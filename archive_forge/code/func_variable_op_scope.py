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
@tf_export(v1=['variable_op_scope'])
@tf_contextlib.contextmanager
def variable_op_scope(values, name_or_scope, default_name=None, initializer=None, regularizer=None, caching_device=None, partitioner=None, custom_getter=None, reuse=None, dtype=None, use_resource=None, constraint=None):
    """Deprecated: context manager for defining an op that creates variables."""
    logging.warn('tf.variable_op_scope(values, name, default_name) is deprecated, use tf.variable_scope(name, default_name, values)')
    with variable_scope(name_or_scope, default_name=default_name, values=values, initializer=initializer, regularizer=regularizer, caching_device=caching_device, partitioner=partitioner, custom_getter=custom_getter, reuse=reuse, dtype=dtype, use_resource=use_resource, constraint=constraint) as scope:
        yield scope