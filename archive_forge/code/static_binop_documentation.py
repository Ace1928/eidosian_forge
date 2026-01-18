from numba.core import errors, ir
from numba.core.rewrites import register_rewrite, Rewrite

        Store constant arguments that were detected in match().
        