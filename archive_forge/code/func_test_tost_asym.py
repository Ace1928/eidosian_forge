import numpy as np
from numpy.testing import assert_almost_equal, assert_equal, assert_
import pytest
import statsmodels.stats.weightstats as smws
from statsmodels.tools.testing import Holder
def test_tost_asym():
    x1, x2 = (clinic[:15, 2], clinic[15:, 2])
    assert_almost_equal(x2.mean() - x1.mean(), tost_clinic_1_asym.estimate, 13)
    resa = smws.ttost_ind(x2, x1, -1.5, 0.6, usevar='unequal')
    assert_almost_equal(resa[0], tost_clinic_1_asym.p_value, 13)
    resall = smws.ttost_ind(clinic[15:, 2:7], clinic[:15, 2:7], [-1.0, -1.0, -1.5, -1.5, -1.5], 0.6, usevar='unequal')
    assert_almost_equal(resall[0], tost_clinic_all_no_multi.p_value, 13)
    resall = smws.ttost_ind(clinic[15:, 2:7], clinic[:15, 2:7], np.exp([-1.0, -1.0, -1.5, -1.5, -1.5]), 0.6, usevar='unequal', transform=np.log)
    resall = smws.ttost_ind(clinic[15:, 2:7], clinic[:15, 2:7], [-1.0, -1.0, -1.5, -1.5, -1.5], 0.6, usevar='unequal', transform=np.exp)
    resall = smws.ttost_paired(clinic[15:, 2:7], clinic[:15, 2:7], [-1.0, -1.0, -1.5, -1.5, -1.5], 0.6, transform=np.log)
    resall = smws.ttost_paired(clinic[15:, 2:7], clinic[:15, 2:7], [-1.0, -1.0, -1.5, -1.5, -1.5], 0.6, transform=np.exp)
    resall = smws.ttest_ind(clinic[15:, 2:7], clinic[:15, 2:7], value=[-1.0, -1.0, -1.5, -1.5, -1.5])
    resall = smws.ttost_ind(clinic[15:, 2:7], clinic[:15, 2:3], [-1.0, -1.0, -1.5, -1.5, -1.5], 0.6, usevar='unequal')
    resa3_2 = smws.ttost_ind(clinic[15:, 3:4], clinic[:15, 2:3], [-1.0, -1.0, -1.5, -1.5, -1.5], 0.6, usevar='unequal')
    assert_almost_equal(resall[0][1], resa3_2[0][1], decimal=13)
    resall = smws.ttost_ind(clinic[15:, 2], clinic[:15, 2], [-1.0, -0.5, -0.7, -1.5, -1.5], 0.6, usevar='unequal')
    resall = smws.ttost_ind(clinic[15:, 2], clinic[:15, 2], [-1.0, -0.5, -0.7, -1.5, -1.5], np.repeat(0.6, 5), usevar='unequal')