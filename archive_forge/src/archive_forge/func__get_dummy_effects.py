from statsmodels.compat.python import lzip
import numpy as np
from scipy.stats import norm
from statsmodels.tools.decorators import cache_readonly
def _get_dummy_effects(effects, exog, dummy_ind, method, model, params):
    """
    If there's a dummy variable, the predicted difference is taken at
    0 and 1
    """
    for i in dummy_ind:
        exog0 = exog.copy()
        exog0[:, i] = 0
        effect0 = model.predict(params, exog0)
        exog0[:, i] = 1
        effect1 = model.predict(params, exog0)
        if 'ey' in method:
            effect0 = np.log(effect0)
            effect1 = np.log(effect1)
        effects[:, i] = effect1 - effect0
    return effects