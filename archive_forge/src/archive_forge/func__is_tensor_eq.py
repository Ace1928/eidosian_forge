from sympy.matrices.dense import eye, Matrix
from sympy.tensor.tensor import tensor_indices, TensorHead, tensor_heads, \
from sympy.physics.hep.gamma_matrices import GammaMatrix as G, LorentzIndex, \
from sympy import Symbol
def _is_tensor_eq(arg1, arg2):
    arg1 = canon_bp(arg1)
    arg2 = canon_bp(arg2)
    if isinstance(arg1, TensExpr):
        return arg1.equals(arg2)
    elif isinstance(arg2, TensExpr):
        return arg2.equals(arg1)
    return arg1 == arg2