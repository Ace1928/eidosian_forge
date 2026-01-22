from statsmodels.regression.linear_model import GLS
import numpy as np
from statsmodels.base.model import LikelihoodModelResults
from scipy import sparse
class Sem2SLS:
    """
    Two-Stage Least Squares for Simultaneous equations

    Parameters
    ----------
    sys : list
        [endog1, exog1, endog2, exog2,...] It will be of length 2 x M,
        where M is the number of equations endog = exog.
    indep_endog : dict
        A dictionary mapping the equation to the column numbers of the
        the independent endogenous regressors in each equation.
        It is assumed that the system is entered as broken up into
        LHS and RHS. For now, the values of the dict have to be sequences.
        Note that the keys for the equations should be zero-indexed.
    instruments : ndarray
        Array of the exogenous independent variables.

    Notes
    -----
    This is unfinished, and the design should be refactored.
    Estimation is done by brute force and there is no exploitation of
    the structure of the system.
    """

    def __init__(self, sys, indep_endog=None, instruments=None):
        if len(sys) % 2 != 0:
            raise ValueError('sys must be a list of pairs of endogenous and exogenous variables.  Got length %s' % len(sys))
        M = len(sys[1::2])
        self._M = M
        self.endog = sys[::2]
        self.exog = sys[1::2]
        self._K = [np.linalg.matrix_rank(_) for _ in sys[1::2]]
        self.instruments = instruments
        instr_endog = {}
        [instr_endog.setdefault(_, []) for _ in indep_endog.keys()]
        for eq_key in indep_endog:
            for varcol in indep_endog[eq_key]:
                instr_endog[eq_key].append(self.exog[eq_key][:, varcol])
        self._indep_endog = indep_endog
        _col_map = np.cumsum(np.hstack((0, self._K)))
        for eq_key in indep_endog:
            try:
                iter(indep_endog[eq_key])
            except:
                raise TypeError('The values of the indep_exog dict must be iterable. Got type %s for converter %s' % (type(indep_endog[eq_key]), eq_key))
        self.wexog = self.whiten(instr_endog)

    def whiten(self, Y):
        """
        Runs the first stage of the 2SLS.

        Returns the RHS variables that include the instruments.
        """
        wexog = []
        indep_endog = self._indep_endog
        instruments = self.instruments
        for eq in range(self._M):
            instr_eq = Y.get(eq, None)
            newRHS = self.exog[eq].copy()
            if instr_eq:
                for i, LHS in enumerate(instr_eq):
                    yhat = GLS(LHS, self.instruments).fit().fittedvalues
                    newRHS[:, indep_endog[eq][i]] = yhat
            wexog.append(newRHS)
        return wexog

    def fit(self):
        """
        """
        delta = []
        wexog = self.wexog
        endog = self.endog
        for j in range(self._M):
            delta.append(GLS(endog[j], wexog[j]).fit().params)
        return delta