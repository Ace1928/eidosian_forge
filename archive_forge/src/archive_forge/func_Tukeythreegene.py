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
def Tukeythreegene(first, second, third):
    firstmean = np.mean(first)
    secondmean = np.mean(second)
    thirdmean = np.mean(third)
    firststd = np.std(first)
    secondstd = np.std(second)
    thirdstd = np.std(third)
    firsts2 = math.pow(firststd, 2)
    seconds2 = math.pow(secondstd, 2)
    thirds2 = math.pow(thirdstd, 2)
    mserrornum = firsts2 * 2 + seconds2 * 2 + thirds2 * 2
    mserrorden = len(first) + len(second) + len(third) - 3
    mserror = mserrornum / mserrorden
    standarderror = math.sqrt(mserror / len(first))
    dftotal = len(first) + len(second) + len(third) - 1
    dfgroups = 2
    dferror = dftotal - dfgroups
    qcrit = 0.5
    qcrit = get_tukeyQcrit(3, dftotal, alpha=0.05)
    qtest3to1 = math.fabs(thirdmean - firstmean) / standarderror
    qtest3to2 = math.fabs(thirdmean - secondmean) / standarderror
    qtest2to1 = math.fabs(secondmean - firstmean) / standarderror
    conclusion = []
    print(qtest3to1)
    print(qtest3to2)
    print(qtest2to1)
    if qtest3to1 > qcrit:
        conclusion.append('3to1null')
    else:
        conclusion.append('3to1alt')
    if qtest3to2 > qcrit:
        conclusion.append('3to2null')
    else:
        conclusion.append('3to2alt')
    if qtest2to1 > qcrit:
        conclusion.append('2to1null')
    else:
        conclusion.append('2to1alt')
    return conclusion