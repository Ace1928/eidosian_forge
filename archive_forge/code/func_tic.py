import numpy as np
from scipy import stats
def tic(results):
    """Takeuchi information criterion for misspecified models

    """
    imr = getattr(results, 'im_ratio', im_ratio(results))
    tic = -2 * results.llf + 2 * np.trace(imr)
    return tic