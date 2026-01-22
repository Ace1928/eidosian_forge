import os
import numpy as np
from numpy import genfromtxt
class ARIMA211:

    def __init__(self, method='mle'):
        if method == 'mle':
            from .arima111_results import results
            self.__dict__.update(results)
            self.resid = self.resid[1:]
            self.params = self.params[:-1]
            self.sigma2 = self.sigma ** 2
            self.aic = self.icstats[4]
            self.bic = self.icstats[5]
            self.fittedvalues = self.xb[1:]
            self.linear = self.y[1:]
            self.k_diff = 1
            self.arroots = [1.027 + 0j, 5.7255 + 0j]
            self.maroots = [1.1442 + 0j]
            self.hqic = 496.5314
            self.aic_gretl = 489.8388
            self.bic_gretl = 506.3801
            self.tvalues = [3.468, 11.14, -1.941, 12.55]
            self.pvalues = [0.0005, 8.14e-29, 0.0522, 3.91e-36]
            cov_params = np.array([[0.0616906, -0.00250187, 0.0010129, 0.00260485], [0, 0.0105302, -0.00867819, -0.00525614], [0, 0, 0.00759185, 0.00361962], [0, 0, 0, 0.00484898]])
            self.cov_params = cov_params + cov_params.T - np.diag(np.diag(cov_params))
            self.bse = np.sqrt(np.diag(self.cov_params))
            self.forecast = forecast_results['fc211c'][-25:]
            self.forecasterr = forecast_results['fc211cse'][-25:]
            self.forecast_dyn = forecast_results['fc211cdyn'][-25:]
            self.forecasterr_dyn = forecast_results['fc211cdynse'][-25:]
        else:
            from .arima211_css_results import results
            self.__dict__.update(results)
            self.resid = self.resid[1:]
            self.params = self.params[:-1]
            self.sigma2 = self.sigma ** 2
            self.aic = self.icstats[4]
            self.bic = self.icstats[5]
            self.fittedvalues = self.xb[1:]
            self.linear = self.y[1:]
            self.k_diff = 1
            self.arroots = [1.0229 + 0j, 4.4501 + 0j]
            self.maroots = [1.0604 + 0j]
            self.hqic = 489.3225
            self.aic_gretl = 482.6486
            self.bic_gretl = 499.1402
            self.tvalues = [0.7206, 22.54, -19.04]
            self.pvalues = [0.4712, 1.52e-112, 2.19e-10, 8e-81]
            cov_params = np.array([[0.000820496, -0.0011992, 0.000457078, 0.00109907], [0, 0.00284432, -0.0016752, -0.00220223], [0, 0, 0.00119783, 0.00108868], [0, 0, 0, 0.00245324]])
            self.cov_params = cov_params + cov_params.T - np.diag(np.diag(cov_params))
            self.bse = np.sqrt(np.diag(self.cov_params))
            self.forecast = forecast_results['fc111c_css'][-25:]
            self.forecasterr = forecast_results['fc111cse_css'][-25:]
            self.forecast_dyn = forecast_results['fc111cdyn_css']
            self.forecasterr_dyn = forecast_results['fc111cdynse_css']