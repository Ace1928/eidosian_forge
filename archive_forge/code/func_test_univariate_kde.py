from unittest import SkipTest
import numpy as np
from holoviews import Area, Bivariate, Contours, Distribution, Image, Polygons
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation.stats import bivariate_kde, univariate_kde
def test_univariate_kde(self):
    kde = univariate_kde(self.dist, n_samples=5, bin_range=(0, 4))
    xs = np.arange(5)
    ys = [0.17594505, 0.23548218, 0.23548218, 0.17594505, 0.0740306]
    area = Area((xs, ys), 'Value', ('Value_density', 'Density'))
    self.assertEqual(kde, area)