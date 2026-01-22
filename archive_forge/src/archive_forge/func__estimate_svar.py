import numpy as np
import numpy.linalg as npl
from numpy.linalg import slogdet
from statsmodels.tools.decorators import deprecated_alias
from statsmodels.tools.numdiff import approx_fprime, approx_hess
import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.tsa.vector_ar.irf import IRAnalysis
import statsmodels.tsa.vector_ar.util as util
from statsmodels.tsa.vector_ar.var_model import VARProcess, VARResults
def _estimate_svar(self, start_params, lags, maxiter, maxfun, trend='c', solver='nm', override=False):
    """
        lags : int
        trend : {str, None}
            As per above
        """
    k_trend = util.get_trendorder(trend)
    y = self.endog
    z = util.get_var_endog(y, lags, trend=trend, has_constant='raise')
    y_sample = y[lags:]
    var_params = np.linalg.lstsq(z, y_sample, rcond=-1)[0]
    resid = y_sample - np.dot(z, var_params)
    avobs = len(y_sample)
    df_resid = avobs - (self.neqs * lags + k_trend)
    sse = np.dot(resid.T, resid)
    omega = sse / df_resid
    self.sigma_u = omega
    A, B = self._solve_AB(start_params, override=override, solver=solver, maxiter=maxiter)
    A_mask = self.A_mask
    B_mask = self.B_mask
    return SVARResults(y, z, var_params, omega, lags, names=self.endog_names, trend=trend, dates=self.data.dates, model=self, A=A, B=B, A_mask=A_mask, B_mask=B_mask)