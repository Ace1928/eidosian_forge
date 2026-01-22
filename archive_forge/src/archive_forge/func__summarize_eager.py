from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.ops import gen_logging_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util import tf_should_use
from tensorflow.python.util.tf_export import tf_export
def _summarize_eager(tensor, summarize=None):
    """Returns a summarized string representation of eager `tensor`.

  Args:
    tensor: EagerTensor to summarize
    summarize: Include these many first elements of `array`
  """
    if summarize is None:
        summarize = 3
    elif summarize < 0:
        summarize = array_ops.size(tensor)
    if tensor._rank():
        flat = tensor.numpy().reshape((-1,))
        lst = [str(x) for x in flat[:summarize]]
        if len(lst) < flat.size:
            lst.append('...')
    elif gen_math_ops.not_equal(summarize, 0):
        lst = [str(tensor.numpy())]
    else:
        lst = []
    return ', '.join(lst)