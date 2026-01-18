from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_addition
from tensorflow.python.ops.linalg import linear_operator_algebra
from tensorflow.python.ops.linalg import linear_operator_block_diag
from tensorflow.python.ops.linalg import linear_operator_block_lower_triangular
from tensorflow.python.ops.linalg import linear_operator_circulant
from tensorflow.python.ops.linalg import linear_operator_diag
from tensorflow.python.ops.linalg import linear_operator_full_matrix
from tensorflow.python.ops.linalg import linear_operator_householder
from tensorflow.python.ops.linalg import linear_operator_identity
from tensorflow.python.ops.linalg import linear_operator_inversion
from tensorflow.python.ops.linalg import linear_operator_kronecker
Inverse of LinearOperatorBlockLowerTriangular.

  We recursively apply the identity:

  ```none
  |A 0|'  =  |    A'  0|
  |B C|      |-C'BA' C'|
  ```

  where `A` is n-by-n, `B` is m-by-n, `C` is m-by-m, and `'` denotes inverse.

  This identity can be verified through multiplication:

  ```none
  |A 0||    A'  0|
  |B C||-C'BA' C'|

    = |       AA'   0|
      |BA'-CC'BA' CC'|

    = |I 0|
      |0 I|
  ```

  Args:
    block_lower_triangular_operator: Instance of
      `LinearOperatorBlockLowerTriangular`.

  Returns:
    block_lower_triangular_operator_inverse: Instance of
      `LinearOperatorBlockLowerTriangular`, the inverse of
      `block_lower_triangular_operator`.
  