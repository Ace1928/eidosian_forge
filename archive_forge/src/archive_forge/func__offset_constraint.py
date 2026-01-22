import numpy as np
from scipy.stats import f as fdist
from scipy.stats import t as student_t
from scipy import stats
from statsmodels.tools.tools import clean0, fullrank
from statsmodels.stats.multitest import multipletests
def _offset_constraint(r_matrix, params_est, params_alt):
    """offset to the value of a linear constraint for new params

    usage:
    (cm, v) is original constraint

    vo = offset_constraint(cm, res2.params, params_alt)
    fs = res2.wald_test((cm, v + vo))
    nc = fs.statistic * fs.df_num

    """
    diff_est = r_matrix @ params_est
    diff_alt = r_matrix @ params_alt
    return diff_est - diff_alt