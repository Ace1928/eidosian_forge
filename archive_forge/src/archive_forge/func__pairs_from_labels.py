from statsmodels.compat.pandas import Appender
from collections import defaultdict
import warnings
import numpy as np
import pandas as pd
from scipy import linalg as spl
from statsmodels.stats.correlation_tools import cov_nearest
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.validation import bool_like
def _pairs_from_labels(self):
    from collections import defaultdict
    pairs = defaultdict(lambda: defaultdict(lambda: None))
    model = self.model
    df = pd.DataFrame({'labels': self.labels, 'groups': model.groups})
    gb = df.groupby(['groups', 'labels'])
    ulabels = np.unique(self.labels)
    for g_ix, g_lb in enumerate(model.group_labels):
        for lx1 in range(len(ulabels)):
            for lx2 in range(lx1 + 1):
                lb1 = ulabels[lx1]
                lb2 = ulabels[lx2]
                try:
                    i1 = gb.groups[g_lb, lb1]
                    i2 = gb.groups[g_lb, lb2]
                except KeyError:
                    continue
                i1, i2 = self._make_pairs(i1, i2)
                clabel = str(lb1) + '/' + str(lb2)
                jj = np.flatnonzero(i1 == i2)
                if len(jj) > 0:
                    clabelv = clabel + '/v'
                    pairs[g_lb][clabelv] = (i1[jj], i2[jj])
                jj = np.flatnonzero(i1 != i2)
                if len(jj) > 0:
                    i1 = i1[jj]
                    i2 = i2[jj]
                    pairs[g_lb][clabel] = (i1, i2)
    self.pairs = pairs