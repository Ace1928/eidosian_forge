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
class GroupsStats:
    """
    statistics by groups (another version)

    groupstats as a class with lazy evaluation (not yet - decorators are still
    missing)

    written this time as equivalent of scipy.stats.rankdata
    gs = GroupsStats(X, useranks=True)
    assert_almost_equal(gs.groupmeanfilter, stats.rankdata(X[:,0]), 15)

    TODO: incomplete doc strings

    """

    def __init__(self, x, useranks=False, uni=None, intlab=None):
        """descriptive statistics by groups

        Parameters
        ----------
        x : ndarray, 2d
            first column data, second column group labels
        useranks : bool
            if true, then use ranks as data corresponding to the
            scipy.stats.rankdata definition (start at 1, ties get mean)
        uni, intlab : arrays (optional)
            to avoid call to unique, these can be given as inputs


        """
        self.x = np.asarray(x)
        if intlab is None:
            uni, intlab = np.unique(x[:, 1], return_inverse=True)
        elif uni is None:
            uni = np.unique(x[:, 1])
        self.useranks = useranks
        self.uni = uni
        self.intlab = intlab
        self.groupnobs = groupnobs = np.bincount(intlab)
        self.runbasic(useranks=useranks)

    def runbasic_old(self, useranks=False):
        """runbasic_old"""
        x = self.x
        if useranks:
            self.xx = x[:, 1].argsort().argsort() + 1
        else:
            self.xx = x[:, 0]
        self.groupsum = groupranksum = np.bincount(self.intlab, weights=self.xx)
        self.groupmean = grouprankmean = groupranksum * 1.0 / self.groupnobs
        self.groupmeanfilter = grouprankmean[self.intlab]

    def runbasic(self, useranks=False):
        """runbasic"""
        x = self.x
        if useranks:
            xuni, xintlab = np.unique(x[:, 0], return_inverse=True)
            ranksraw = x[:, 0].argsort().argsort() + 1
            self.xx = GroupsStats(np.column_stack([ranksraw, xintlab]), useranks=False).groupmeanfilter
        else:
            self.xx = x[:, 0]
        self.groupsum = groupranksum = np.bincount(self.intlab, weights=self.xx)
        self.groupmean = grouprankmean = groupranksum * 1.0 / self.groupnobs
        self.groupmeanfilter = grouprankmean[self.intlab]

    def groupdemean(self):
        """groupdemean"""
        return self.xx - self.groupmeanfilter

    def groupsswithin(self):
        """groupsswithin"""
        xtmp = self.groupdemean()
        return np.bincount(self.intlab, weights=xtmp ** 2)

    def groupvarwithin(self):
        """groupvarwithin"""
        return self.groupsswithin() / (self.groupnobs - 1)