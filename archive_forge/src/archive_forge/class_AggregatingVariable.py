import contextlib
import copy
import functools
import threading
import weakref
import numpy as np
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import values
from tensorflow.python.distribute import values_util
from tensorflow.python.distribute.coordinator import coordinator_context
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.saved_model import save_context
from tensorflow.python.trackable import base as trackable
from tensorflow.python.types import core
class AggregatingVariable(resource_variable_ops.BaseResourceVariable, core.Tensor):
    """A wrapper around a variable that aggregates updates across replicas."""

    def __init__(self, strategy, v, aggregation):
        self._distribute_strategy = strategy
        self._v = v
        v._aggregating_container = weakref.ref(self)
        self._aggregation = aggregation

    def __deepcopy__(self, memo):
        """Perform a deepcopy of the `AggregatingVariable`.

    Unlike the deepcopy of a regular tf.Variable, this keeps the original
    strategy and devices of the `AggregatingVariable`.  To avoid confusion
    with the behavior of deepcopy on a regular `Variable` (which does
    copy into new devices), we only allow a deepcopy of a `AggregatingVariable`
    within its originating strategy scope.

    Args:
      memo: The memoization object for `deepcopy`.

    Returns:
      A deep copy of the current `AggregatingVariable`.

    Raises:
      RuntimeError: If trying to deepcopy into a different strategy.
    """
        with distribute_lib.enter_or_assert_strategy(self._distribute_strategy):
            v = copy.deepcopy(self._v, memo)
        copied_variable = type(self)(strategy=self._distribute_strategy, v=v, aggregation=self._aggregation)
        memo[id(self)] = copied_variable
        return copied_variable

    def get(self):
        return self._v

    @property
    def distribute_strategy(self):
        return self._distribute_strategy

    def __getattr__(self, name):
        return getattr(self._v, name)

    def _assign_func(self, *args, **kwargs):
        with distribute_lib.enter_or_assert_strategy(self._distribute_strategy):
            f = kwargs.pop('f')
            if distribute_lib.in_cross_replica_context():
                if distribute_lib.get_update_replica_id() is not None:
                    return f(self._v, *args, **kwargs)
                return self._distribute_strategy.extended.update(self, f, args=args, kwargs=kwargs)
            else:
                replica_context = distribute_lib.get_replica_context()
                assert replica_context
                if self._aggregation == vs.VariableAggregation.NONE:
                    raise ValueError(values_util.aggregation_error_msg.format(variable_type='AggregatingVariable'))

                def merge_fn(strategy, value, use_locking=False, name=None, read_value=True):
                    v = values_util.apply_aggregation(strategy, value, self._aggregation, self)
                    if name and isinstance(name, values.PerReplica):
                        name = name.values[0]
                    return strategy.extended.update(self, f, args=(v,), kwargs={'use_locking': use_locking, 'name': name, 'read_value': read_value})
                return replica_context.merge_call(merge_fn, args=args, kwargs=kwargs)

    def assign_sub(self, *args, **kwargs):
        assign_sub_fn = lambda var, *a, **kw: var.assign_sub(*a, **kw)
        return self._assign_func(*args, f=assign_sub_fn, **kwargs)

    def assign_add(self, *args, **kwargs):
        assign_add_fn = lambda var, *a, **kw: var.assign_add(*a, **kw)
        return self._assign_func(*args, f=assign_add_fn, **kwargs)

    def assign(self, *args, **kwargs):
        assign_fn = lambda var, *a, **kw: var.assign(*a, **kw)
        return self._assign_func(*args, f=assign_fn, **kwargs)

    @property
    def initializer(self):
        return self._v.initializer

    def initialized_value(self):
        return self._v.initialized_value()

    @property
    def initial_value(self):
        return self._v.initial_value

    @property
    def op(self):
        return self._v.op

    def value(self):
        return self._v.value()

    def read_value(self):
        return self._v.read_value()

    def sparse_read(self, indices, name=None):
        return self._v.sparse_read(indices, name=name)

    def eval(self, session=None):
        return self._v.eval(session)

    @property
    def graph(self):
        return self._v.graph

    @property
    def device(self):
        return self._v.device

    @property
    def shape(self):
        return self._v.shape

    @property
    def aggregation(self):
        return self._aggregation

    @property
    def synchronization(self):
        return self._v.synchronization

    @property
    def name(self):
        return self._v.name

    @property
    def trainable(self):
        return self._v.trainable

    @property
    def dtype(self):
        return self._v.dtype

    def _gather_saveables_for_checkpoint(self):
        if isinstance(self._v, CachingVariable):
            return self._v._gather_saveables_for_checkpoint()
        return {trackable.VARIABLE_VALUE_KEY: self._v}

    def _export_to_saved_model_graph(self, object_map, tensor_map, options, **kwargs):
        """For implementing `Trackable`."""
        resource_list = self._v._export_to_saved_model_graph(object_map, tensor_map, options, **kwargs)
        object_map[self] = object_map[self._v]
        return resource_list

    def __add__(self, o):
        return self._v + o

    def __radd__(self, o):
        return o + self._v

    def __sub__(self, o):
        return self._v - o

    def __rsub__(self, o):
        return o - self._v

    def __mul__(self, o):
        return self._v * o

    def __rmul__(self, o):
        return o * self._v

    def __truediv__(self, o):
        return self._v / o

    def __rtruediv__(self, o):
        return o / self._v

    def __floordiv__(self, o):
        return self._v // o

    def __rfloordiv__(self, o):
        return o // self._v

    def __mod__(self, o):
        return self._v % o

    def __rmod__(self, o):
        return o % self._v

    def __lt__(self, o):
        return self._v < o

    def __le__(self, o):
        return self._v <= o

    def __gt__(self, o):
        return self._v > o

    def __ge__(self, o):
        return self._v >= o

    def __and__(self, o):
        return self._v & o

    def __rand__(self, o):
        return o & self._v

    def __or__(self, o):
        return self._v | o

    def __ror__(self, o):
        return o | self._v

    def __xor__(self, o):
        return self._v ^ o

    def __rxor__(self, o):
        return o ^ self._v

    def __getitem__(self, o):
        return self._v[o]

    def __pow__(self, o, modulo=None):
        return pow(self._v, o, modulo)

    def __rpow__(self, o):
        return pow(o, self._v)

    def __invert__(self):
        return ~self._v

    def __neg__(self):
        return -self._v

    def __abs__(self):
        return abs(self._v)

    def __div__(self, o):
        try:
            return self._v.__div__(o)
        except AttributeError:
            return NotImplemented

    def __rdiv__(self, o):
        try:
            return self._v.__rdiv__(o)
        except AttributeError:
            return NotImplemented

    def __matmul__(self, o):
        try:
            return self._v.__matmul__(o)
        except AttributeError:
            return NotImplemented

    def __rmatmul__(self, o):
        try:
            return self._v.__rmatmul__(o)
        except AttributeError:
            return NotImplemented

    def __str__(self):
        return str(self._v)

    def __repr__(self):
        return repr(self._v)

    def _should_act_as_resource_variable(self):
        """Pass resource_variable_ops.is_resource_variable check."""
        pass

    def _dense_var_to_tensor(self, dtype=None, name=None, as_ref=False):
        return self._v._dense_var_to_tensor(dtype=dtype, name=name, as_ref=as_ref)