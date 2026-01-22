import numpy as np
import scipy.stats
import warnings
def _link_deprecation_warning(old, new):
    warnings.warn(f'The {old} link alias is deprecated. Use {new} instead. The {old} link alias will be removed after the 0.15.0 release.', FutureWarning)