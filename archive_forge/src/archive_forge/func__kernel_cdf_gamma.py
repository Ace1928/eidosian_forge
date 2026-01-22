import numpy as np
from scipy import special, stats
def _kernel_cdf_gamma(x, sample, bw):
    """Gamma kernel for cdf, without boundary corrected part.

    drops `+ 1` in shape parameter

    It should be possible to use this if probability in
    neighborhood of zero boundary is small.

    """
    return stats.gamma.sf(sample, x / bw, scale=bw)