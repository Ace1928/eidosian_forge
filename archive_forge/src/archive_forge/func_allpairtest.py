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
def allpairtest(self, testfunc, alpha=0.05, method='bonf', pvalidx=1):
    """run a pairwise test on all pairs with multiple test correction

        The statistical test given in testfunc is calculated for all pairs
        and the p-values are adjusted by methods in multipletests. The p-value
        correction is generic and based only on the p-values, and does not
        take any special structure of the hypotheses into account.

        Parameters
        ----------
        testfunc : function
            A test function for two (independent) samples. It is assumed that
            the return value on position pvalidx is the p-value.
        alpha : float
            familywise error rate
        method : str
            This specifies the method for the p-value correction. Any method
            of multipletests is possible.
        pvalidx : int (default: 1)
            position of the p-value in the return of testfunc

        Returns
        -------
        sumtab : SimpleTable instance
            summary table for printing

        errors:  TODO: check if this is still wrong, I think it's fixed.
        results from multipletests are in different order
        pval_corrected can be larger than 1 ???
        """
    res = []
    for i, j in zip(*self.pairindices):
        res.append(testfunc(self.datali[i], self.datali[j]))
    res = np.array(res)
    reject, pvals_corrected, alphacSidak, alphacBonf = multipletests(res[:, pvalidx], alpha=alpha, method=method)
    i1, i2 = self.pairindices
    if pvals_corrected is None:
        resarr = np.array(lzip(self.groupsunique[i1], self.groupsunique[i2], np.round(res[:, 0], 4), np.round(res[:, 1], 4), reject), dtype=[('group1', object), ('group2', object), ('stat', float), ('pval', float), ('reject', np.bool_)])
    else:
        resarr = np.array(lzip(self.groupsunique[i1], self.groupsunique[i2], np.round(res[:, 0], 4), np.round(res[:, 1], 4), np.round(pvals_corrected, 4), reject), dtype=[('group1', object), ('group2', object), ('stat', float), ('pval', float), ('pval_corr', float), ('reject', np.bool_)])
    results_table = SimpleTable(resarr, headers=resarr.dtype.names)
    results_table.title = 'Test Multiple Comparison %s \n%s%4.2f method=%s' % (testfunc.__name__, 'FWER=', alpha, method) + '\nalphacSidak=%4.2f, alphacBonf=%5.3f' % (alphacSidak, alphacBonf)
    return (results_table, (res, reject, pvals_corrected, alphacSidak, alphacBonf), resarr)