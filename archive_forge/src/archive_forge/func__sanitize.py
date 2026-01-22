from tensorflow.python.autograph.core import ag_ctx
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.operators import variables
from tensorflow.python.framework import auto_control_deps
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.util import nest
def _sanitize(self, name):
    """See https://www.tensorflow.org/api_docs/python/tf/Graph#name_scope."""
    if name and name.startswith('_'):
        name = 'fn' + name
    return name