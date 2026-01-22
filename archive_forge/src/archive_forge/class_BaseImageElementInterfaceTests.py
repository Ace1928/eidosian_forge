import datetime as dt
from unittest import SkipTest
import numpy as np
from holoviews import HSV, RGB, Curve, Dataset, Dimension, Image, Table
from holoviews.core.data.interface import DataError
from holoviews.core.util import date_range
from .base import DatatypeContext, GriddedInterfaceTests, InterfaceTests
class BaseImageElementInterfaceTests(InterfaceTests):
    """
    Tests for ImageInterface
    """
    element = Image
    __test__ = False

    def init_grid_data(self):
        self.xs = np.linspace(-9, 9, 10)
        self.ys = np.linspace(0.5, 9.5, 10)
        self.array = np.arange(10) * np.arange(10)[:, np.newaxis]

    def init_data(self):
        self.image = Image(np.flipud(self.array), bounds=(-10, 0, 10, 10))

    def test_init_data_tuple(self):
        xs = np.arange(5)
        ys = np.arange(10)
        array = xs * ys[:, np.newaxis]
        Image((xs, ys, array))

    def test_init_data_tuple_error(self):
        xs = np.arange(5)
        ys = np.arange(10)
        array = xs * ys[:, np.newaxis]
        with self.assertRaises(DataError):
            Image((ys, xs, array))

    def test_bounds_mismatch(self):
        with self.assertRaises(ValueError):
            Image((range(10), range(10), np.random.rand(10, 10)), bounds=0.5)

    def test_init_data_datetime_xaxis(self):
        start = np.datetime64(dt.datetime.today())
        end = start + np.timedelta64(1, 's')
        xs = date_range(start, end, 10).astype('datetime64[ns]')
        Image((xs, self.ys, self.array))

    def test_init_data_datetime_yaxis(self):
        start = np.datetime64(dt.datetime.today())
        end = start + np.timedelta64(1, 's')
        ys = date_range(start, end, 10).astype('datetime64[ns]')
        Image((self.xs, ys, self.array))

    def test_init_bounds(self):
        self.assertEqual(self.image.bounds.lbrt(), (-10, 0, 10, 10))

    def test_init_bounds_datetime_xaxis(self):
        start = np.datetime64(dt.datetime.today())
        end = start + np.timedelta64(1, 's')
        xs = date_range(start, end, 10).astype('datetime64[ns]')
        bounds = (start, 0, end, 10)
        image = Image((xs, self.ys, self.array), bounds=bounds)
        self.assertEqual(image.bounds.lbrt(), bounds)

    def test_init_bounds_datetime_yaxis(self):
        start = np.datetime64(dt.datetime.today())
        end = start + np.timedelta64(1, 's')
        ys = date_range(start, end, 10).astype('datetime64[ns]')
        bounds = (-10, start, 10, end)
        image = Image((self.xs, ys, self.array))
        self.assertEqual(image.bounds.lbrt(), bounds)

    def test_init_densities(self):
        self.assertEqual(self.image.xdensity, 0.5)
        self.assertEqual(self.image.ydensity, 1)

    def test_init_densities_datetime_xaxis(self):
        start = np.datetime64(dt.datetime.today())
        end = start + np.timedelta64(1, 's')
        xs = date_range(start, end, 10).astype('datetime64[ns]')
        image = Image((xs, self.ys, self.array))
        self.assertEqual(image.xdensity, 1e-05)
        self.assertEqual(image.ydensity, 1)

    def test_init_densities_datetime_yaxis(self):
        start = np.datetime64(dt.datetime.today())
        end = start + np.timedelta64(1, 's')
        ys = date_range(start, end, 10).astype('datetime64[ns]')
        image = Image((self.xs, ys, self.array))
        self.assertEqual(image.xdensity, 0.5)
        self.assertEqual(image.ydensity, 1e-05)

    def test_dimension_values_xs(self):
        self.assertEqual(self.image.dimension_values(0, expanded=False), np.linspace(-9, 9, 10))

    def test_dimension_values_ys(self):
        self.assertEqual(self.image.dimension_values(1, expanded=False), np.linspace(0.5, 9.5, 10))

    def test_dimension_values_vdim(self):
        self.assertEqual(self.image.dimension_values(2, flat=False), self.array)

    def test_index_single_coordinate(self):
        self.assertEqual(self.image[0.3, 5.1], 25)

    def test_slice_xaxis(self):
        sliced = self.image[0.3:5.2]
        self.assertEqual(sliced.bounds.lbrt(), (0, 0, 6, 10))
        self.assertEqual(sliced.xdensity, 0.5)
        self.assertEqual(sliced.ydensity, 1)
        self.assertEqual(sliced.dimension_values(2, flat=False), self.array[:, 5:8])

    def test_slice_datetime_xaxis(self):
        start = np.datetime64(dt.datetime.today())
        end = start + np.timedelta64(1, 's')
        bounds = (start, 0, end, 10)
        xs = date_range(start, end, 10).astype('datetime64[ns]')
        image = Image((xs, self.ys, self.array), bounds=bounds)
        sliced = image[start + np.timedelta64(530, 'ms'):start + np.timedelta64(770, 'ms')]
        self.assertEqual(sliced.dimension_values(2, flat=False), self.array[:, 5:8])

    def test_slice_yaxis(self):
        sliced = self.image[:, 1.2:5.2]
        self.assertEqual(sliced.bounds.lbrt(), (-10, 1.0, 10, 5))
        self.assertEqual(sliced.xdensity, 0.5)
        self.assertEqual(sliced.ydensity, 1)
        self.assertEqual(sliced.dimension_values(2, flat=False), self.array[1:5, :])

    def test_slice_datetime_yaxis(self):
        start = np.datetime64(dt.datetime.today())
        end = start + np.timedelta64(1, 's')
        ys = date_range(start, end, 10).astype('datetime64[ns]')
        image = Image((self.xs, ys, self.array))
        sliced = image[:, start + np.timedelta64(120, 'ms'):start + np.timedelta64(520, 'ms')]
        self.assertEqual(sliced.dimension_values(2, flat=False), self.array[1:5, :])

    def test_slice_both_axes(self):
        sliced = self.image[0.3:5.2, 1.2:5.2]
        self.assertEqual(sliced.bounds.lbrt(), (0, 1.0, 6, 5))
        self.assertEqual(sliced.xdensity, 0.5)
        self.assertEqual(sliced.ydensity, 1)
        self.assertEqual(sliced.dimension_values(2, flat=False), self.array[1:5, 5:8])

    def test_slice_x_index_y(self):
        sliced = self.image[0.3:5.2, 5.2]
        self.assertEqual(sliced.bounds.lbrt(), (0, 5.0, 6.0, 6.0))
        self.assertEqual(sliced.xdensity, 0.5)
        self.assertEqual(sliced.ydensity, 1)
        self.assertEqual(sliced.dimension_values(2, flat=False), self.array[5:6, 5:8])

    def test_index_x_slice_y(self):
        sliced = self.image[3.2, 1.2:5.2]
        self.assertEqual(sliced.bounds.lbrt(), (2.0, 1.0, 4.0, 5.0))
        self.assertEqual(sliced.xdensity, 0.5)
        self.assertEqual(sliced.ydensity, 1)
        self.assertEqual(sliced.dimension_values(2, flat=False), self.array[1:5, 6:7])

    def test_range_xdim(self):
        self.assertEqual(self.image.range(0), (-10, 10))

    def test_range_datetime_xdim(self):
        start = np.datetime64(dt.datetime.today())
        end = start + np.timedelta64(1, 's')
        xs = date_range(start, end, 10).astype('datetime64[ns]')
        image = Image((xs, self.ys, self.array))
        self.assertEqual(image.range(0), (start, end))

    def test_range_ydim(self):
        self.assertEqual(self.image.range(1), (0, 10))

    def test_range_datetime_ydim(self):
        start = np.datetime64(dt.datetime.today())
        end = start + np.timedelta64(1, 's')
        ys = date_range(start, end, 10).astype('datetime64[ns]')
        image = Image((self.xs, ys, self.array))
        self.assertEqual(image.range(1), (start, end))

    def test_range_vdim(self):
        self.assertEqual(self.image.range(2), (0, 81))

    def test_dimension_values_xcoords(self):
        self.assertEqual(self.image.dimension_values(0, expanded=False), np.linspace(-9, 9, 10))

    def test_dimension_values_datetime_xcoords(self):
        start = np.datetime64(dt.datetime.today())
        end = start + np.timedelta64(1, 's')
        xs = date_range(start, end, 10).astype('datetime64[ns]')
        image = Image((xs, self.ys, self.array))
        self.assertEqual(image.dimension_values(0, expanded=False), date_range(start, end, 10))

    def test_dimension_values_ycoords(self):
        self.assertEqual(self.image.dimension_values(1, expanded=False), np.linspace(0.5, 9.5, 10))

    def test_sample_xcoord(self):
        ys = np.linspace(0.5, 9.5, 10)
        zs = [0, 7, 14, 21, 28, 35, 42, 49, 56, 63]
        with DatatypeContext([self.datatype, 'dictionary', 'dataframe'], self.image):
            self.assertEqual(self.image.sample(x=5), Curve((ys, zs), kdims=['y'], vdims=['z']))

    def test_sample_ycoord(self):
        xs = np.linspace(-9, 9, 10)
        zs = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36]
        with DatatypeContext([self.datatype, 'dictionary', 'dataframe'], self.image):
            self.assertEqual(self.image.sample(y=5), Curve((xs, zs), kdims=['x'], vdims=['z']))

    def test_sample_coords(self):
        arr = np.arange(10) * np.arange(5)[np.newaxis].T
        xs = np.linspace(0.12, 0.81, 10)
        ys = np.linspace(0.12, 0.391, 5)
        img = Image((xs, ys, arr), kdims=['x', 'y'], vdims=['z'], datatype=[self.datatype])
        sampled = img.sample([(0.15, 0.15), (0.15, 0.4), (0.8, 0.4), (0.8, 0.15)])
        self.assertIsInstance(sampled, Table)
        yidx = [0, 4, 4, 0]
        xidx = [0, 0, 9, 9]
        table = Table((xs[xidx], ys[yidx], arr[yidx, xidx]), kdims=['x', 'y'], vdims=['z'])
        self.assertEqual(sampled, table)

    def test_reduce_to_scalar(self):
        self.assertEqual(self.image.reduce(['x', 'y'], function=np.mean), 20.25)

    def test_reduce_x_dimension(self):
        ys = np.linspace(0.5, 9.5, 10)
        zs = [0.0, 4.5, 9.0, 13.5, 18.0, 22.5, 27.0, 31.5, 36.0, 40.5]
        with DatatypeContext([self.datatype, 'dictionary', 'dataframe'], Image):
            self.assertEqual(self.image.reduce(x=np.mean), Curve((ys, zs), kdims=['y'], vdims=['z']))

    def test_reduce_y_dimension(self):
        xs = np.linspace(-9, 9, 10)
        zs = [0.0, 4.5, 9.0, 13.5, 18.0, 22.5, 27.0, 31.5, 36.0, 40.5]
        with DatatypeContext([self.datatype, 'dictionary', 'dataframe'], Image):
            self.assertEqual(self.image.reduce(y=np.mean), Curve((xs, zs), kdims=['x'], vdims=['z']))

    def test_dataset_reindex_constant(self):
        with DatatypeContext([self.datatype, 'dictionary', 'dataframe', 'grid'], self.image):
            selected = Dataset(self.image.select(x=0))
            reindexed = selected.reindex(['y'])
        data = Dataset(selected.columns(['y', 'z']), kdims=['y'], vdims=['z'])
        self.assertEqual(reindexed, data)

    def test_dataset_reindex_non_constant(self):
        with DatatypeContext([self.datatype, 'dictionary', 'dataframe', 'grid'], self.image):
            ds = Dataset(self.image)
            reindexed = ds.reindex(['y'])
        data = Dataset(ds.columns(['y', 'z']), kdims=['y'], vdims=['z'])
        self.assertEqual(reindexed, data)

    def test_aggregate_with_spreadfn(self):
        with DatatypeContext([self.datatype, 'dictionary', 'dataframe'], self.image):
            agg = self.image.aggregate('x', np.mean, np.std)
        xs = self.image.dimension_values('x', expanded=False)
        mean = self.array.mean(axis=0)
        std = self.array.std(axis=0)
        self.assertEqual(agg, Curve((xs, mean, std), kdims=['x'], vdims=['z', 'z_std']))