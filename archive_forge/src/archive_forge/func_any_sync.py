from .decorators import jit
import numba
@jit(device=True)
def any_sync(mask, predicate):
    """
    If for any thread in the masked warp the predicate is true, then
    a non-zero value is returned, otherwise 0 is returned.
    """
    return numba.cuda.vote_sync_intrinsic(mask, 1, predicate)[1]