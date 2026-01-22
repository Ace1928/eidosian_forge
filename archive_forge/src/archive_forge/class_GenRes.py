import numpy as np
from numpy.testing import assert_almost_equal
from statsmodels.datasets import star98
from statsmodels.emplike.descriptive import DescStat
from .results.el_results import DescStatRes
class GenRes:
    """
    Reads in the data and creates class instance to be tested
    """

    @classmethod
    def setup_class(cls):
        data = star98.load()
        data.exog = np.asarray(data.exog)
        desc_stat_data = data.exog[:50, 5]
        mv_desc_stat_data = data.exog[:50, 5:7]
        cls.res1 = DescStat(desc_stat_data)
        cls.res2 = DescStatRes()
        cls.mvres1 = DescStat(mv_desc_stat_data)