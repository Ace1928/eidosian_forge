from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import manip_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.util.tf_export import tf_export
def _construct_adjoint_diagonals(self, diagonals):
    if self.diagonals_format == _SEQUENCE:
        diagonals = [math_ops.conj(d) for d in reversed(diagonals)]
        diagonals[0] = manip_ops.roll(diagonals[0], shift=-1, axis=-1)
        diagonals[2] = manip_ops.roll(diagonals[2], shift=1, axis=-1)
        return diagonals
    elif self.diagonals_format == _MATRIX:
        return linalg.adjoint(diagonals)
    else:
        diagonals = math_ops.conj(diagonals)
        superdiag, diag, subdiag = array_ops_stack.unstack(diagonals, num=3, axis=-2)
        new_superdiag = manip_ops.roll(subdiag, shift=-1, axis=-1)
        new_subdiag = manip_ops.roll(superdiag, shift=1, axis=-1)
        return array_ops_stack.stack([new_superdiag, diag, new_subdiag], axis=-2)