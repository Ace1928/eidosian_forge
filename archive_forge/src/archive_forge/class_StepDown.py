from pystatsmodels mailinglist 20100524
from collections import namedtuple
from statsmodels.compat.python import lzip, lrange
import copy
import math
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
from scipy import stats, interpolate
from statsmodels.iolib.table import SimpleTable
from statsmodels.stats.multitest import multipletests, _ecdf as ecdf, fdrcorrection as fdrcorrection0, fdrcorrection_twostage
from statsmodels.graphics import utils
from statsmodels.tools.sm_exceptions import ValueWarning
class StepDown:
    """a class for step down methods

    This is currently for simple tree subset descend, similar to homogeneous_subsets,
    but checks all leave-one-out subsets instead of assuming an ordered set.
    Comment in SAS manual:
    SAS only uses interval subsets of the sorted list, which is sufficient for range
    tests (maybe also equal variance and balanced sample sizes are required).
    For F-test based critical distances, the restriction to intervals is not sufficient.

    This version uses a single critical value of the studentized range distribution
    for all comparisons, and is therefore a step-down version of Tukey HSD.
    The class is written so it can be subclassed, where the get_distance_matrix and
    get_crit are overwritten to obtain other step-down procedures such as REGW.

    iter_subsets can be overwritten, to get a recursion as in the many to one comparison
    with a control such as in Dunnet's test.


    A one-sided right tail test is not covered because the direction of the inequality
    is hard coded in check_set.  Also Peritz's check of partitions is not possible, but
    I have not seen it mentioned in any more recent references.
    I have only partially read the step-down procedure for closed tests by Westfall.

    One change to make it more flexible, is to separate out the decision on a subset,
    also because the F-based tests, FREGW in SPSS, take information from all elements of
    a set and not just pairwise comparisons. I have not looked at the details of
    the F-based tests such as Sheffe yet. It looks like running an F-test on equality
    of means in each subset. This would also outsource how pairwise conditions are
    combined, any larger or max. This would also imply that the distance matrix cannot
    be calculated in advance for tests like the F-based ones.


    """

    def __init__(self, vals, nobs_all, var_all, df=None):
        self.vals = vals
        self.n_vals = len(vals)
        self.nobs_all = nobs_all
        self.var_all = var_all
        self.df = df

    def get_crit(self, alpha):
        """
        get_tukeyQcrit

        currently tukey Q, add others
        """
        q_crit = get_tukeyQcrit(self.n_vals, self.df, alpha=alpha)
        return q_crit * np.ones(self.n_vals)

    def get_distance_matrix(self):
        """studentized range statistic"""
        dres = distance_st_range(self.vals, self.nobs_all, self.var_all, df=self.df)
        self.distance_matrix = dres[0]

    def iter_subsets(self, indices):
        """Iterate substeps"""
        for ii in range(len(indices)):
            idxsub = copy.copy(indices)
            idxsub.pop(ii)
            yield idxsub

    def check_set(self, indices):
        """check whether pairwise distances of indices satisfy condition

        """
        indtup = tuple(indices)
        if indtup in self.cache_result:
            return self.cache_result[indtup]
        else:
            set_distance_matrix = self.distance_matrix[np.asarray(indices)[:, None], indices]
            n_elements = len(indices)
            if np.any(set_distance_matrix > self.crit[n_elements - 1]):
                res = True
            else:
                res = False
            self.cache_result[indtup] = res
            return res

    def stepdown(self, indices):
        """stepdown"""
        print(indices)
        if self.check_set(indices):
            if len(indices) > 2:
                for subs in self.iter_subsets(indices):
                    self.stepdown(subs)
            else:
                self.rejected.append(tuple(indices))
        else:
            self.accepted.append(tuple(indices))
            return indices

    def run(self, alpha):
        """main function to run the test,

        could be done in __call__ instead
        this could have all the initialization code

        """
        self.cache_result = {}
        self.crit = self.get_crit(alpha)
        self.accepted = []
        self.rejected = []
        self.get_distance_matrix()
        self.stepdown(lrange(self.n_vals))
        return (list(set(self.accepted)), list(set(sd.rejected)))