from numba.core import errors, ir, consts
from numba.core.rewrites import register_rewrite, Rewrite
def _try_infer_constant(self, func_ir, inst):
    try:
        return func_ir.infer_constant(inst.exception)
    except consts.ConstantInferenceError:
        return None