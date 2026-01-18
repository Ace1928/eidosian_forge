import json
import os
import numpy as np
from numpy.testing import (
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
from statsmodels.datasets import macrodata, sunspots
from statsmodels.regression.linear_model import OLS
import statsmodels.stats.diagnostic as smsdia
import statsmodels.stats.outliers_influence as oi
import statsmodels.stats.sandwich_covariance as sw
from statsmodels.tools.tools import Bunch, add_constant
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
def test_outlier_test():
    endog, exog, labels = get_duncan_data()
    ndarray_mod = OLS(endog, exog).fit()
    rstudent = [3.1345185839, -2.397022399, 2.0438046359, -1.9309187757, 1.8870465798, -1.76049053, -1.7040324156, 1.6024285876, -1.4332485037, -1.1044851583, 1.0688582315, 1.018527184, -0.9024219332, -0.9023876471, -0.8830953936, 0.8265782334, 0.8089220547, 0.7682770197, 0.7319491074, -0.6665962829, 0.5227352794, -0.5135016547, 0.5083881518, 0.4999224372, -0.4980818221, -0.4759717075, -0.429356582, -0.4114056499, -0.3779540862, 0.355687403, 0.3409200462, 0.3062248646, 0.3038999429, -0.3030815773, -0.1873387893, 0.1738050251, 0.1424246593, -0.1292266025, 0.1272066463, -0.0798902878, 0.0788467222, 0.0722556991, 0.050509828, 0.0233215136, 0.0007112055]
    unadj_p = [0.003177202, 0.021170298, 0.047432955, 0.060427645, 0.06624812, 0.085783008, 0.095943909, 0.116738318, 0.15936889, 0.275822623, 0.291386358, 0.314400295, 0.372104049, 0.37212204, 0.382333561, 0.413260793, 0.423229432, 0.44672537, 0.468363101, 0.508764039, 0.60397199, 0.610356737, 0.613905871, 0.619802317, 0.621087703, 0.636621083, 0.669911674, 0.682917818, 0.707414459, 0.723898263, 0.734904667, 0.760983108, 0.762741124, 0.763360242, 0.852319039, 0.862874018, 0.887442197, 0.897810225, 0.899398691, 0.936713197, 0.937538115, 0.942749758, 0.959961394, 0.981506948, 0.999435989]
    bonf_p = [0.1429741, 0.9526634, 2.134483, 2.719244, 2.9811654, 3.8602354, 4.3174759, 5.2532243, 7.1716001, 12.412018, 13.1123861, 14.1480133, 16.7446822, 16.7454918, 17.2050103, 18.5967357, 19.0453245, 20.1026416, 21.0763395, 22.8943818, 27.1787396, 27.4660532, 27.6257642, 27.8911043, 27.9489466, 28.6479487, 30.1460253, 30.7313018, 31.8336506, 32.5754218, 33.07071, 34.2442399, 34.3233506, 34.3512109, 38.3543568, 38.8293308, 39.9348989, 40.4014601, 40.4729411, 42.1520939, 42.1892152, 42.4237391, 43.1982627, 44.1678127, 44.9746195]
    bonf_p = np.array(bonf_p)
    bonf_p[bonf_p > 1] = 1
    sorted_labels = ['minister', 'reporter', 'contractor', 'insurance.agent', 'machinist', 'store.clerk', 'conductor', 'factory.owner', 'mail.carrier', 'streetcar.motorman', 'carpenter', 'coal.miner', 'bartender', 'bookkeeper', 'soda.clerk', 'chemist', 'RR.engineer', 'professor', 'electrician', 'gas.stn.attendant', 'auto.repairman', 'watchman', 'banker', 'machine.operator', 'dentist', 'waiter', 'shoe.shiner', 'welfare.worker', 'plumber', 'physician', 'pilot', 'engineer', 'accountant', 'lawyer', 'undertaker', 'barber', 'store.manager', 'truck.driver', 'cook', 'janitor', 'policeman', 'architect', 'teacher', 'taxi.driver', 'author']
    res2 = np.c_[rstudent, unadj_p, bonf_p]
    res = oi.outlier_test(ndarray_mod, method='b', labels=labels, order=True)
    np.testing.assert_almost_equal(res.values, res2, 7)
    np.testing.assert_equal(res.index.tolist(), sorted_labels)
    data = pd.DataFrame(np.column_stack((endog, exog)), columns='y const var1 var2'.split(), index=labels)
    res_pd = OLS.from_formula('y ~ const + var1 + var2 - 0', data).fit()
    res_outl2 = oi.outlier_test(res_pd, method='b', order=True)
    assert_almost_equal(res_outl2.values, res2, 7)
    assert_equal(res_outl2.index.tolist(), sorted_labels)
    res_outl1 = res_pd.outlier_test(method='b')
    res_outl1 = res_outl1.sort_values(['unadj_p'], ascending=True)
    assert_almost_equal(res_outl1.values, res2, 7)
    assert_equal(res_outl1.index.tolist(), sorted_labels)
    assert_array_equal(res_outl2.index, res_outl1.index)
    res_outl3 = res_pd.outlier_test(method='b', order=True)
    assert_equal(res_outl3.index.tolist(), sorted_labels)
    res_outl4 = res_pd.outlier_test(method='b', order=True, cutoff=0.15)
    assert_equal(res_outl4.index.tolist(), sorted_labels[:1])