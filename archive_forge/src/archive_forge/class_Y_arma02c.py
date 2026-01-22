import os
from numpy import genfromtxt
class Y_arma02c:

    def __init__(self, method='mle'):
        if method == 'mle':
            self.params = [4.519277801954, 0.20038540396, -0.643766305844]
            self.aic = 758.05119454077
            self.bic = 772.137038212219
            self.arroots = None
            self.maroots = [-1.1004 + 0j, 1.4117 + 0j]
            self.bse = [0.038397713362, 0.049314652466, 0.048961366071]
            self.cov_params = [[0.0014744, 6.2363e-05, 6.4093e-05], [6.2363e-05, 0.0024319, 0.0014083], [6.4093e-05, 0.0014083, 0.0023972]]
            self.hqic = 763.720334415218
            self.llf = -375.025597270385
            self.resid = residsc_mle[:, 5]
            self.fittedvalues = yhatc_mle[:, 5]
            self.pvalues = [0.0, 4.84e-05, 1.74e-39]
            self.tvalues = [117.7, 4.063, -13.15]
            self.sigma2 = 1.081406299967 ** 2
        elif method == 'css':
            self.params = [4.519869870853, 0.202414429306, -0.647482560461]
            self.aic = 756.679105324347
            self.bic = 770.764948995796
            self.arroots = None
            self.maroots = [-1.0962 + 0j, 1.4089 + 0j]
            self.bse = [0.038411589816, 0.047983057239, 0.043400749866]
            self.cov_params = [[0.00146121526606, 5.30770136338e-05, 5.34796521051e-05], [5.30770136338e-05, 0.00237105883909, 0.00141090983316], [5.34796521051e-05, 0.00141090983316, 0.0023558435508]]
            self.hqic = 762.348245198795
            self.llf = -374.339552662174
            self.resid = residsc_css[:, 5]
            self.fittedvalues = yhatc_css[:, 5]
            self.pvalues = [0.0, 2.46e-05, 2.49e-50]
            self.tvalues = [118.24120637494, 4.15691796413, -13.33981086206]
            self.sigma2 = 1.081576475937 ** 2