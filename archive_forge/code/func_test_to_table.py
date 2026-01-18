import numpy as np
from numpy.testing import assert_almost_equal, assert_equal, assert_allclose
from statsmodels.stats.inter_rater import (fleiss_kappa, cohens_kappa,
from statsmodels.tools.testing import Holder
def test_to_table():
    data = diagnoses
    res1 = to_table(data[:, :2] - 1, 5)
    res0 = np.asarray([[(data[:, :2] - 1 == [i, j]).all(1).sum() for j in range(5)] for i in range(5)])
    assert_equal(res1[0], res0)
    res2 = to_table(data[:, :2])
    assert_equal(res2[0], res0)
    bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    res3 = to_table(data[:, :2], bins)
    assert_equal(res3[0], res0)
    res4 = to_table(data[:, :3] - 1, bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5])
    res5 = to_table(data[:, :3] - 1, bins=5)
    assert_equal(res4[0].sum(-1), res0)
    assert_equal(res5[0].sum(-1), res0)