from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_tpu_ops
from tensorflow.python.ops.gen_tpu_ops import *
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import tpu_function
from tensorflow.python.util.tf_export import tf_export
@ops.RegisterGradient('AllToAll')
def _all_to_all_grad(op, grad):
    return [gen_tpu_ops.all_to_all(grad, op.inputs[1], concat_dimension=op.get_attr('split_dimension'), split_dimension=op.get_attr('concat_dimension'), split_count=op.get_attr('split_count')), None]