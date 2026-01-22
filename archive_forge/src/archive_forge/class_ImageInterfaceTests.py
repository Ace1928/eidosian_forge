import datetime as dt
from unittest import SkipTest
import numpy as np
from holoviews import HSV, RGB, Curve, Dataset, Dimension, Image, Table
from holoviews.core.data.interface import DataError
from holoviews.core.util import date_range
from .base import DatatypeContext, GriddedInterfaceTests, InterfaceTests
class ImageInterfaceTests(GriddedInterfaceTests, InterfaceTests):
    """
    Tests for ImageInterface
    """
    datatype = 'image'
    data_type = np.ndarray
    element = Image
    __test__ = True

    def test_canonical_vdim(self):
        x = np.array([0.0, 0.75, 1.5])
        y = np.array([1.5, 0.75, 0.0])
        z = np.array([[0.06925999, 0.05800389, 0.05620127], [0.06240918, 0.05800931, 0.04969735], [0.05376789, 0.04669417, 0.03880118]])
        dataset = Image((x, y, z), kdims=['x', 'y'], vdims=['z'])
        canonical = np.array([[0.05376789, 0.04669417, 0.03880118], [0.06240918, 0.05800931, 0.04969735], [0.06925999, 0.05800389, 0.05620127]])
        self.assertEqual(dataset.dimension_values('z', flat=False), canonical)

    def test_gridded_dtypes(self):
        ds = self.dataset_grid
        self.assertEqual(ds.interface.dtype(ds, 'x'), np.float64)
        self.assertEqual(ds.interface.dtype(ds, 'y'), np.float64)
        self.assertEqual(ds.interface.dtype(ds, 'z'), np.dtype(int))

    def test_dataset_groupby_with_transposed_dimensions(self):
        raise SkipTest('Image interface does not support multi-dimensional data.')

    def test_dataset_dynamic_groupby_with_transposed_dimensions(self):
        raise SkipTest('Image interface does not support multi-dimensional data.')

    def test_dataset_slice_inverted_dimension(self):
        raise SkipTest('Image interface does not support 1D data')

    def test_sample_2d(self):
        raise SkipTest('Image interface only supports Image type')