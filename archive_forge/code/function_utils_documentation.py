import functools
import six
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
Returns func_code of passed callable, or None if not available.