from collections import defaultdict
import numpy as np
from statsmodels.base._penalties import SCADSmoothed
def _get_penal(self, weights=None):
    """create new Penalty instance
        """
    return SCADSmoothed(0.1, c0=0.0001, weights=weights)