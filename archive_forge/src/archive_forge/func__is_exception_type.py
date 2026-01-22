from numba.core import errors, ir, consts
from numba.core.rewrites import register_rewrite, Rewrite
def _is_exception_type(self, const):
    return isinstance(const, type) and issubclass(const, Exception)