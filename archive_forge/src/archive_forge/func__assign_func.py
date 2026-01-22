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