from functools import reduce
import numpy as np
from statsmodels.regression.linear_model import GLS
from pandas import Panel
class PanelModel:
    """
    An abstract statistical model class for panel (longitudinal) datasets.

    Parameters
    ----------
    endog : array_like or str
        If a pandas object is used then endog should be the name of the
        endogenous variable as a string.
#    exog
#    panel_arr
#    time_arr
    panel_data : pandas.Panel object

    Notes
    -----
    If a pandas object is supplied it is assumed that the major_axis is time
    and that the minor_axis has the panel variable.
    """

    def __init__(self, endog=None, exog=None, panel=None, time=None, xtnames=None, equation=None, panel_data=None):
        if panel_data is None:
            self.initialize(endog, exog, panel, time, xtnames, equation)

    def initialize(self, endog, exog, panel, time, xtnames, equation):
        """
        Initialize plain array model.

        See PanelModel
        """
        names = equation.split(' ')
        self.endog_name = names[0]
        exog_names = names[1:]
        self.panel_name = xtnames[0]
        self.time_name = xtnames[1]
        novar = exog.var(0) == 0
        if True in novar:
            cons_index = np.where(novar == 1)[0][0]
            exog_names.insert(cons_index, 'cons')
        self._cons_index = novar
        self.exog_names = exog_names
        self.endog = np.squeeze(np.asarray(endog))
        exog = np.asarray(exog)
        self.exog = exog
        self.panel = np.asarray(panel)
        self.time = np.asarray(time)
        self.paneluniq = np.unique(panel)
        self.timeuniq = np.unique(time)

    def initialize_pandas(self, panel_data, endog_name, exog_name):
        self.panel_data = panel_data
        endog = panel_data[endog_name].values
        self.endog = np.squeeze(endog)
        if exog_name is None:
            exog_name = panel_data.columns.tolist()
            exog_name.remove(endog_name)
        self.exog = panel_data.filterItems(exog_name).values
        self._exog_name = exog_name
        self._endog_name = endog_name
        self._timeseries = panel_data.major_axis
        self._panelseries = panel_data.minor_axis

    def _group_mean(self, X, index='oneway', counts=False, dummies=False):
        """
        Get group means of X by time or by panel.

        index default is panel
        """
        if index == 'oneway':
            Y = self.panel
            uniq = self.paneluniq
        elif index == 'time':
            Y = self.time
            uniq = self.timeuniq
        else:
            raise ValueError('index %s not understood' % index)
        print(Y, uniq, uniq[:, None], len(Y), len(uniq), len(uniq[:, None]), index)
        dummy = (Y == uniq[:, None]).astype(float)
        if X.ndim > 1:
            mean = np.dot(dummy, X) / dummy.sum(1)[:, None]
        else:
            mean = np.dot(dummy, X) / dummy.sum(1)
        if counts is False and dummies is False:
            return mean
        elif counts is True and dummies is False:
            return (mean, dummy.sum(1))
        elif counts is True and dummies is True:
            return (mean, dummy.sum(1), dummy)
        elif counts is False and dummies is True:
            return (mean, dummy)

    def fit(self, model=None, method=None, effects='oneway'):
        """
        method : LSDV, demeaned, MLE, GLS, BE, FE, optional
        model :
                between
                fixed
                random
                pooled
                [gmm]
        effects :
                oneway
                time
                twoway
        femethod : demeaned (only one implemented)
                   WLS
        remethod :
                swar -
                amemiya
                nerlove
                walhus


        Notes
        -----
        This is unfinished.  None of the method arguments work yet.
        Only oneway effects should work.
        """
        if method:
            method = method.lower()
        model = model.lower()
        if method and method not in ['lsdv', 'demeaned', 'mle', 'gls', 'be', 'fe']:
            raise ValueError('%s not a valid method' % method)
        if model == 'pooled':
            return GLS(self.endog, self.exog).fit()
        if model == 'between':
            return self._fit_btwn(method, effects)
        if model == 'fixed':
            return self._fit_fixed(method, effects)

    def _fit_btwn(self, method, effects):
        if effects != 'twoway':
            endog = self._group_mean(self.endog, index=effects)
            exog = self._group_mean(self.exog, index=effects)
        else:
            raise ValueError('%s effects is not valid for the between estimator' % effects)
        befit = GLS(endog, exog).fit()
        return befit

    def _fit_fixed(self, method, effects):
        endog = self.endog
        exog = self.exog
        demeantwice = False
        if effects in ['oneway', 'twoways']:
            if effects == 'twoways':
                demeantwice = True
                effects = 'oneway'
            endog_mean, counts = self._group_mean(endog, index=effects, counts=True)
            exog_mean = self._group_mean(exog, index=effects)
            counts = counts.astype(int)
            endog = endog - np.repeat(endog_mean, counts)
            exog = exog - np.repeat(exog_mean, counts, axis=0)
        if demeantwice or effects == 'time':
            endog_mean, dummies = self._group_mean(endog, index='time', dummies=True)
            exog_mean = self._group_mean(exog, index='time')
            endog = endog - np.dot(endog_mean, dummies)
            exog = exog - np.dot(dummies.T, exog_mean)
        fefit = GLS(endog, exog[:, -self._cons_index]).fit()
        return fefit