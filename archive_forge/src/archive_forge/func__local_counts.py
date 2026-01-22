import numpy as np
from scipy.special import comb
def _local_counts(args):
    mnc, mnc2, mnc3, mnc4 = args
    mc = mnc
    mc2 = mnc2 - mnc * mnc
    mc3 = mnc3 - (3 * mc * mc2 + mc ** 3)
    mc4 = mnc4 - (4 * mc * mc3 + 6 * mc * mc * mc2 + mc ** 4)
    return mc2mvsk((mc, mc2, mc3, mc4))