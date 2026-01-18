from .decorators import jit
import numba

    Shuffles value across the masked warp and returns the value
    from (laneid ^ lane_mask).
    