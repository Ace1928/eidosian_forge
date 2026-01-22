import numpy as np
from scipy import stats  #get rid of this? need only norm.sf
class KappaResults(ResultsBunch):
    """Results for Cohen's kappa

    Attributes
    ----------
    kappa : cohen's kappa
    var_kappa : variance of kappa
    std_kappa : standard deviation of kappa
    alpha : one-sided probability for confidence interval
    kappa_low : lower (1-alpha) confidence limit
    kappa_upp : upper (1-alpha) confidence limit
    var_kappa0 : variance of kappa under H0: kappa=0
    std_kappa0 : standard deviation of kappa under H0: kappa=0
    z_value : test statistic for H0: kappa=0, is standard normal distributed
    pvalue_one_sided : one sided p-value for H0: kappa=0 and H1: kappa>0
    pvalue_two_sided : two sided p-value for H0: kappa=0 and H1: kappa!=0
    distribution_kappa : asymptotic normal distribution of kappa
    distribution_zero_null : asymptotic normal distribution of kappa under
        H0: kappa=0

    The confidence interval for kappa and the statistics for the test of
    H0: kappa=0 are based on the asymptotic normal distribution of kappa.

    """
    template = _kappa_template

    def _initialize(self):
        if 'alpha' not in self:
            self['alpha'] = 0.025
            self['alpha_ci'] = _int_ifclose(100 - 0.025 * 200)[1]
        self['std_kappa'] = np.sqrt(self['var_kappa'])
        self['std_kappa0'] = np.sqrt(self['var_kappa0'])
        self['z_value'] = self['kappa'] / self['std_kappa0']
        self['pvalue_one_sided'] = stats.norm.sf(self['z_value'])
        self['pvalue_two_sided'] = stats.norm.sf(np.abs(self['z_value'])) * 2
        delta = stats.norm.isf(self['alpha']) * self['std_kappa']
        self['kappa_low'] = self['kappa'] - delta
        self['kappa_upp'] = self['kappa'] + delta
        self['distribution_kappa'] = stats.norm(loc=self['kappa'], scale=self['std_kappa'])
        self['distribution_zero_null'] = stats.norm(loc=0, scale=self['std_kappa0'])

    def __str__(self):
        return self.template % self