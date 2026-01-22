from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.util import object_identity
class UnliftableError(Exception):
    """Raised if a Tensor cannot be lifted from the graph."""
    ag_pass_through = True