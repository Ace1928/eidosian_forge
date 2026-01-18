import copy
import enum
import functools
import sys
import threading
import traceback
from tensorflow.python import tf2
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import core
from tensorflow.python.util import deprecation
from tensorflow.python.util import function_utils
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['variable_creator_scope'])
@tf_contextlib.contextmanager
def variable_creator_scope_v1(variable_creator):
    """Scope which defines a variable creation function to be used by variable().

  variable_creator is expected to be a function with the following signature:

  ```
    def variable_creator(next_creator, **kwargs)
  ```

  The creator is supposed to eventually call the next_creator to create a
  variable if it does want to create a variable and not call Variable or
  ResourceVariable directly. This helps make creators composable. A creator may
  choose to create multiple variables, return already existing variables, or
  simply register that a variable was created and defer to the next creators in
  line. Creators can also modify the keyword arguments seen by the next
  creators.

  Custom getters in the variable scope will eventually resolve down to these
  custom creators when they do create variables.

  The valid keyword arguments in kwds are:

   * initial_value: A `Tensor`, or Python object convertible to a `Tensor`,
        which is the initial value for the Variable. The initial value must have
        a shape specified unless `validate_shape` is set to False. Can also be a
        callable with no argument that returns the initial value when called. In
        that case, `dtype` must be specified. (Note that initializer functions
        from init_ops.py must first be bound to a shape before being used here.)
   * trainable: If `True`, the default, also adds the variable to the graph
        collection `GraphKeys.TRAINABLE_VARIABLES`. This collection is used as
        the default list of variables to use by the `Optimizer` classes.
        `trainable` defaults to `True`, unless `synchronization` is
        set to `ON_READ`, in which case it defaults to `False`.
   * collections: List of graph collections keys. The new variable is added to
        these collections. Defaults to `[GraphKeys.GLOBAL_VARIABLES]`.
   * validate_shape: If `False`, allows the variable to be initialized with a
        value of unknown shape. If `True`, the default, the shape of
        `initial_value` must be known.
   * caching_device: Optional device string describing where the Variable
        should be cached for reading.  Defaults to the Variable's device.
        If not `None`, caches on another device.  Typical use is to cache
        on the device where the Ops using the Variable reside, to deduplicate
        copying through `Switch` and other conditional statements.
   * name: Optional name for the variable. Defaults to `'Variable'` and gets
        uniquified automatically.
   * dtype: If set, initial_value will be converted to the given type.
        If `None`, either the datatype will be kept (if `initial_value` is
        a Tensor), or `convert_to_tensor` will decide.
   * constraint: A constraint function to be applied to the variable after
        updates by some algorithms.
   * use_resource: if True, a ResourceVariable is always created.
   * synchronization: Indicates when a distributed a variable will be
        aggregated. Accepted values are constants defined in the class
        `tf.VariableSynchronization`. By default the synchronization is set to
        `AUTO` and the current `DistributionStrategy` chooses
        when to synchronize.
   * aggregation: Indicates how a distributed variable will be aggregated.
        Accepted values are constants defined in the class
        `tf.VariableAggregation`.

  This set may grow over time, so it's important the signature of creators is as
  mentioned above.

  Args:
    variable_creator: the passed creator

  Yields:
    A scope in which the creator is active
  """
    with ops.get_default_graph()._variable_creator_scope(variable_creator):
        yield