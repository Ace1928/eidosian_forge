import numpy as np
from statsmodels.base import model
import statsmodels.base.model as base
from statsmodels.tools.decorators import cache_readonly
from statsmodels.compat.pandas import Appender
class PHSurvivalTime:

    def __init__(self, time, status, exog, strata=None, entry=None, offset=None):
        """
        Represent a collection of survival times with possible
        stratification and left truncation.

        Parameters
        ----------
        time : array_like
            The times at which either the event (failure) occurs or
            the observation is censored.
        status : array_like
            Indicates whether the event (failure) occurs at `time`
            (`status` is 1), or if `time` is a censoring time (`status`
            is 0).
        exog : array_like
            The exogeneous (covariate) data matrix, cases are rows and
            variables are columns.
        strata : array_like
            Grouping variable defining the strata.  If None, all
            observations are in a single stratum.
        entry : array_like
            Entry (left truncation) times.  The observation is not
            part of the risk set for times before the entry time.  If
            None, the entry time is treated as being zero, which
            gives no left truncation.  The entry time must be less
            than or equal to `time`.
        offset : array_like
            An optional array of offsets
        """
        if strata is None:
            strata = np.zeros(len(time), dtype=np.int32)
        if entry is None:
            entry = np.zeros(len(time))
        self._check(time, status, strata, entry)
        stu = np.unique(strata)
        sth = {x: [] for x in stu}
        for i, k in enumerate(strata):
            sth[k].append(i)
        stratum_rows = [np.asarray(sth[k], dtype=np.int32) for k in stu]
        stratum_names = stu
        ix = [i for i, ix in enumerate(stratum_rows) if status[ix].sum() > 0]
        self.nstrat_orig = len(stratum_rows)
        stratum_rows = [stratum_rows[i] for i in ix]
        stratum_names = [stratum_names[i] for i in ix]
        nstrat = len(stratum_rows)
        self.nstrat = nstrat
        for stx, ix in enumerate(stratum_rows):
            last_failure = max(time[ix][status[ix] == 1])
            ii = [i for i, t in enumerate(entry[ix]) if t <= last_failure]
            stratum_rows[stx] = stratum_rows[stx][ii]
        for stx, ix in enumerate(stratum_rows):
            first_failure = min(time[ix][status[ix] == 1])
            ii = [i for i, t in enumerate(time[ix]) if t >= first_failure]
            stratum_rows[stx] = stratum_rows[stx][ii]
        for stx, ix in enumerate(stratum_rows):
            ii = np.argsort(time[ix])
            stratum_rows[stx] = stratum_rows[stx][ii]
        if offset is not None:
            self.offset_s = []
            for stx in range(nstrat):
                self.offset_s.append(offset[stratum_rows[stx]])
        else:
            self.offset_s = None
        self.n_obs = sum([len(ix) for ix in stratum_rows])
        self.stratum_rows = stratum_rows
        self.stratum_names = stratum_names
        self.time_s = self._split(time)
        self.exog_s = self._split(exog)
        self.status_s = self._split(status)
        self.entry_s = self._split(entry)
        self.ufailt_ix, self.risk_enter, self.risk_exit, self.ufailt = ([], [], [], [])
        for stx in range(self.nstrat):
            ift = np.flatnonzero(self.status_s[stx] == 1)
            ft = self.time_s[stx][ift]
            uft = np.unique(ft)
            nuft = len(uft)
            uft_map = {x: i for i, x in enumerate(uft)}
            uft_ix = [[] for k in range(nuft)]
            for ix, ti in zip(ift, ft):
                uft_ix[uft_map[ti]].append(ix)
            risk_enter1 = [[] for k in range(nuft)]
            for i, t in enumerate(self.time_s[stx]):
                ix = np.searchsorted(uft, t, 'right') - 1
                if ix >= 0:
                    risk_enter1[ix].append(i)
            risk_exit1 = [[] for k in range(nuft)]
            for i, t in enumerate(self.entry_s[stx]):
                ix = np.searchsorted(uft, t)
                risk_exit1[ix].append(i)
            self.ufailt.append(uft)
            self.ufailt_ix.append([np.asarray(x, dtype=np.int32) for x in uft_ix])
            self.risk_enter.append([np.asarray(x, dtype=np.int32) for x in risk_enter1])
            self.risk_exit.append([np.asarray(x, dtype=np.int32) for x in risk_exit1])

    def _split(self, x):
        v = []
        if x.ndim == 1:
            for ix in self.stratum_rows:
                v.append(x[ix])
        else:
            for ix in self.stratum_rows:
                v.append(x[ix, :])
        return v

    def _check(self, time, status, strata, entry):
        n1, n2, n3, n4 = (len(time), len(status), len(strata), len(entry))
        nv = [n1, n2, n3, n4]
        if max(nv) != min(nv):
            raise ValueError('endog, status, strata, and ' + 'entry must all have the same length')
        if min(time) < 0:
            raise ValueError('endog must be non-negative')
        if min(entry) < 0:
            raise ValueError('entry time must be non-negative')
        if np.any(entry > time):
            raise ValueError('entry times may not occur ' + 'after event or censoring times')