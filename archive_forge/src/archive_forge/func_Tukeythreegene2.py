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
def Tukeythreegene2(genes):
    """gend is a list, ie [first, second, third]"""
    means = []
    stds = []
    for gene in genes:
        means.append(np.mean(gene))
        std.append(np.std(gene))
    stds2 = []
    for std in stds:
        stds2.append(math.pow(std, 2))
    mserrornum = sum(stds2) * 2
    mserrorden = len(genes[0]) + len(genes[1]) + len(genes[2]) - 3
    mserror = mserrornum / mserrorden