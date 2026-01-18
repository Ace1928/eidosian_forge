import re
from tensorflow.python import tf2
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.trackable import autotrackable
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util.tf_export import tf_export
def valid_identifier(name):
    return bool(_VALID_IDENTIFIER.match(name))