import datetime as dt
from itertools import product
from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews.core.data import Dataset
from holoviews.core.data.interface import DataError
from holoviews.core.util import date_range
from holoviews.element import HSV, RGB, Curve, Image
from holoviews.util.transform import dim
from .base import (
from .test_imageinterface import (
class BaseGridInterfaceTests(GriddedInterfaceTests, HomogeneousColumnTests, InterfaceTests):
    __test__ = False

    def test_dataset_dataframe_init_hm(self):
        with self.assertRaises(DataError):
            Dataset(pd.DataFrame({'x': self.xs, 'x2': self.xs_2}), kdims=['x'], vdims=['x2'])

    def test_dataset_dataframe_init_hm_alias(self):
        with self.assertRaises(DataError):
            Dataset(pd.DataFrame({'x': self.xs, 'x2': self.xs_2}), kdims=['x'], vdims=['x2'])

    def test_dataset_empty_constructor(self):
        ds = Dataset([], ['x', 'y'], ['z'])
        assert ds.interface.shape(ds, gridded=True) == (0, 0)

    def test_dataset_multi_vdim_empty_constructor(self):
        ds = Dataset([], ['x', 'y'], ['z1', 'z2', 'z3'])
        assert all((ds.dimension_values(vd, flat=False).shape == (0, 0) for vd in ds.vdims))

    def test_irregular_grid_data_values(self):
        nx, ny = (20, 5)
        xs, ys = np.meshgrid(np.arange(nx) + 0.5, np.arange(ny) + 0.5)
        zs = np.arange(100).reshape(5, 20)
        ds = Dataset((xs, ys, zs), ['x', 'y'], 'z')
        self.assertEqual(ds.dimension_values(2, flat=False), zs)
        self.assertEqual(ds.interface.coords(ds, 'x'), xs)
        self.assertEqual(ds.interface.coords(ds, 'y'), ys)

    def test_irregular_grid_data_values_inverted_y(self):
        nx, ny = (20, 5)
        xs, ys = np.meshgrid(np.arange(nx) + 0.5, np.arange(ny) * -1 + 0.5)
        zs = np.arange(100).reshape(5, 20)
        ds = Dataset((xs, ys, zs), ['x', 'y'], 'z')
        self.assertEqual(ds.dimension_values(2, flat=False), zs)
        self.assertEqual(ds.interface.coords(ds, 'x'), xs)
        self.assertEqual(ds.interface.coords(ds, 'y'), ys)

    def test_dataset_sort_hm(self):
        raise SkipTest('Not supported')

    def test_dataset_sort_reverse_hm(self):
        raise SkipTest('Not supported')

    def test_dataset_sort_vdim_hm(self):
        exception = 'Compressed format cannot be sorted, either instantiate in the desired order or use the expanded format.'
        with self.assertRaisesRegex(Exception, exception):
            self.dataset_hm.sort('y')

    def test_dataset_sort_reverse_vdim_hm(self):
        exception = 'Compressed format cannot be sorted, either instantiate in the desired order or use the expanded format.'
        with self.assertRaisesRegex(Exception, exception):
            self.dataset_hm.sort('y', reverse=True)

    def test_dataset_sort_vdim_hm_alias(self):
        exception = 'Compressed format cannot be sorted, either instantiate in the desired order or use the expanded format.'
        with self.assertRaisesRegex(Exception, exception):
            self.dataset_hm.sort('y')

    def test_dataset_groupby(self):
        self.assertEqual(self.dataset_hm.groupby('x').keys(), list(self.xs))

    def test_dataset_add_dimensions_value_hm(self):
        with self.assertRaisesRegex(Exception, 'Cannot add key dimension to a dense representation.'):
            self.dataset_hm.add_dimension('z', 1, 0)

    def test_dataset_add_dimensions_values_hm(self):
        table = self.dataset_hm.add_dimension('z', 1, range(1, 12), vdim=True)
        self.assertEqual(table.vdims[1], 'z')
        self.compare_arrays(table.dimension_values('z'), np.array(list(range(1, 12))))

    def test_dataset_add_dimensions_values_hm_alias(self):
        table = self.dataset_hm.add_dimension(('z', 'Z'), 1, range(1, 12), vdim=True)
        self.assertEqual(table.vdims[1], 'Z')
        self.compare_arrays(table.dimension_values('Z'), np.array(list(range(1, 12))))

    def test_dataset_2D_columnar_shape(self):
        array = np.random.rand(11, 11)
        dataset = Dataset({'x': self.xs, 'y': self.y_ints, 'z': array}, kdims=['x', 'y'], vdims=['z'])
        self.assertEqual(dataset.shape, (11 * 11, 3))

    def test_dataset_2D_gridded_shape(self):
        array = np.random.rand(12, 11)
        dataset = Dataset({'x': self.xs, 'y': range(12), 'z': array}, kdims=['x', 'y'], vdims=['z'])
        self.assertEqual(dataset.interface.shape(dataset, gridded=True), (12, 11))

    def test_dataset_2D_aggregate_partial_hm(self):
        array = np.random.rand(11, 11)
        dataset = Dataset({'x': self.xs, 'y': self.y_ints, 'z': array}, kdims=['x', 'y'], vdims=['z'])
        self.assertEqual(dataset.aggregate(['x'], np.mean), Dataset({'x': self.xs, 'z': np.mean(array, axis=0)}, kdims=['x'], vdims=['z']))

    def test_dataset_2D_aggregate_partial_hm_alias(self):
        array = np.random.rand(11, 11)
        dataset = Dataset({'x': self.xs, 'y': self.y_ints, 'z': array}, kdims=[('x', 'X'), ('y', 'Y')], vdims=[('z', 'Z')])
        self.assertEqual(dataset.aggregate(['X'], np.mean), Dataset({'x': self.xs, 'z': np.mean(array, axis=0)}, kdims=[('x', 'X')], vdims=[('z', 'Z')]))

    def test_dataset_2D_reduce_hm(self):
        array = np.random.rand(11, 11)
        dataset = Dataset({'x': self.xs, 'y': self.y_ints, 'z': array}, kdims=['x', 'y'], vdims=['z'])
        self.assertEqual(np.array(dataset.reduce(['x', 'y'], np.mean)), np.mean(array))

    def test_dataset_2D_reduce_hm_alias(self):
        array = np.random.rand(11, 11)
        dataset = Dataset({'x': self.xs, 'y': self.y_ints, 'z': array}, kdims=[('x', 'X'), ('y', 'Y')], vdims=[('z', 'Z')])
        self.assertEqual(np.array(dataset.reduce(['x', 'y'], np.mean)), np.mean(array))
        self.assertEqual(np.array(dataset.reduce(['X', 'Y'], np.mean)), np.mean(array))

    def test_dataset_groupby_dynamic(self):
        array = np.random.rand(11, 11)
        dataset = Dataset({'x': self.xs, 'y': self.y_ints, 'z': array}, kdims=['x', 'y'], vdims=['z'])
        with DatatypeContext([self.datatype, 'dictionary', 'dataframe'], dataset):
            grouped = dataset.groupby('x', dynamic=True)
        first = Dataset({'y': self.y_ints, 'z': array[:, 0]}, kdims=['y'], vdims=['z'])
        self.assertEqual(grouped[0], first)

    def test_dataset_groupby_dynamic_alias(self):
        array = np.random.rand(11, 11)
        dataset = Dataset({'x': self.xs, 'y': self.y_ints, 'z': array}, kdims=[('x', 'X'), ('y', 'Y')], vdims=[('z', 'Z')])
        with DatatypeContext([self.datatype, 'dictionary', 'dataframe'], dataset):
            grouped = dataset.groupby('X', dynamic=True)
        first = Dataset({'y': self.y_ints, 'z': array[:, 0]}, kdims=[('y', 'Y')], vdims=[('z', 'Z')])
        self.assertEqual(grouped[0], first)

    def test_dataset_groupby_multiple_dims(self):
        dataset = Dataset((range(8), range(8), range(8), range(8), np.random.rand(8, 8, 8, 8)), kdims=['a', 'b', 'c', 'd'], vdims=['Value'])
        grouped = dataset.groupby(['c', 'd'])
        keys = list(product(range(8), range(8)))
        self.assertEqual(list(grouped.keys()), keys)
        for c, d in keys:
            self.assertEqual(grouped[c, d], dataset.select(c=c, d=d).reindex(['a', 'b']))

    def test_dataset_groupby_drop_dims(self):
        array = np.random.rand(3, 20, 10)
        ds = Dataset({'x': range(10), 'y': range(20), 'z': range(3), 'Val': array}, kdims=['x', 'y', 'z'], vdims=['Val'])
        with DatatypeContext([self.datatype, 'dictionary', 'dataframe'], (ds, Dataset)):
            partial = ds.to(Dataset, kdims=['x'], vdims=['Val'], groupby='y')
        self.assertEqual(partial.last['Val'], array[:, -1, :].T.flatten())

    def test_dataset_groupby_drop_dims_dynamic(self):
        array = np.random.rand(3, 20, 10)
        ds = Dataset({'x': range(10), 'y': range(20), 'z': range(3), 'Val': array}, kdims=['x', 'y', 'z'], vdims=['Val'])
        with DatatypeContext([self.datatype, 'dictionary', 'dataframe'], (ds, Dataset)):
            partial = ds.to(Dataset, kdims=['x'], vdims=['Val'], groupby='y', dynamic=True)
            self.assertEqual(partial[19]['Val'], array[:, -1, :].T.flatten())

    def test_dataset_groupby_drop_dims_with_vdim(self):
        array = np.random.rand(3, 20, 10)
        ds = Dataset({'x': range(10), 'y': range(20), 'z': range(3), 'Val': array, 'Val2': array * 2}, kdims=['x', 'y', 'z'], vdims=['Val', 'Val2'])
        with DatatypeContext([self.datatype, 'dictionary', 'dataframe'], (ds, Dataset)):
            partial = ds.to(Dataset, kdims=['Val'], vdims=['Val2'], groupby='y')
        self.assertEqual(partial.last['Val'], array[:, -1, :].T.flatten())

    def test_dataset_groupby_drop_dims_dynamic_with_vdim(self):
        array = np.random.rand(3, 20, 10)
        ds = Dataset({'x': range(10), 'y': range(20), 'z': range(3), 'Val': array, 'Val2': array * 2}, kdims=['x', 'y', 'z'], vdims=['Val', 'Val2'])
        with DatatypeContext([self.datatype, 'dictionary', 'dataframe'], (ds, Dataset)):
            partial = ds.to(Dataset, kdims=['Val'], vdims=['Val2'], groupby='y', dynamic=True)
            self.assertEqual(partial[19]['Val'], array[:, -1, :].T.flatten())

    def test_dataset_ndloc_lists(self):
        xs, ys = (np.linspace(0.12, 0.81, 10), np.linspace(0.12, 0.391, 5))
        arr = np.arange(10) * np.arange(5)[np.newaxis].T
        ds = self.element((xs, ys, arr), kdims=['x', 'y'], vdims=['z'], datatype=[self.datatype, 'dictionary'])
        sliced = self.element((xs[[1, 2, 3]], ys[[0, 1, 2]], arr[[0, 1, 2], [1, 2, 3]]), kdims=['x', 'y'], vdims=['z'], datatype=['dictionary'])
        self.assertEqual(ds.ndloc[[0, 1, 2], [1, 2, 3]], sliced)

    def test_dataset_ndloc_lists_invert_x(self):
        xs, ys = (np.linspace(0.12, 0.81, 10), np.linspace(0.12, 0.391, 5))
        arr = np.arange(10) * np.arange(5)[np.newaxis].T
        ds = self.element((xs[::-1], ys, arr), kdims=['x', 'y'], vdims=['z'], datatype=[self.datatype, 'dictionary'])
        sliced = self.element((xs[::-1][[8, 7, 6]], ys[[0, 1, 2]], arr[[0, 1, 2], [8, 7, 6]]), kdims=['x', 'y'], vdims=['z'], datatype=['dictionary'])
        self.assertEqual(ds.ndloc[[0, 1, 2], [1, 2, 3]], sliced)

    def test_dataset_ndloc_lists_invert_y(self):
        xs, ys = (np.linspace(0.12, 0.81, 10), np.linspace(0.12, 0.391, 5))
        arr = np.arange(10) * np.arange(5)[np.newaxis].T
        ds = self.element((xs, ys[::-1], arr), kdims=['x', 'y'], vdims=['z'], datatype=[self.datatype, 'dictionary'])
        sliced = self.element((xs[[1, 2, 3]], ys[::-1][[4, 3, 2]], arr[[4, 3, 2], [1, 2, 3]]), kdims=['x', 'y'], vdims=['z'], datatype=['dictionary'])
        self.assertEqual(ds.ndloc[[0, 1, 2], [1, 2, 3]], sliced)

    def test_dataset_ndloc_lists_invert_xy(self):
        xs, ys = (np.linspace(0.12, 0.81, 10), np.linspace(0.12, 0.391, 5))
        arr = np.arange(10) * np.arange(5)[np.newaxis].T
        ds = self.element((xs[::-1], ys[::-1], arr), kdims=['x', 'y'], vdims=['z'], datatype=[self.datatype, 'dictionary'])
        sliced = self.element((xs[::-1][[8, 7, 6]], ys[::-1][[4, 3, 2]], arr[[4, 3, 2], [8, 7, 6]]), kdims=['x', 'y'], vdims=['z'], datatype=['dictionary'])
        self.assertEqual(ds.ndloc[[0, 1, 2], [1, 2, 3]], sliced)

    def test_dataset_ndloc_slice_two_vdims(self):
        xs, ys = (np.linspace(0.12, 0.81, 10), np.linspace(0.12, 0.391, 5))
        arr = np.arange(10) * np.arange(5)[np.newaxis].T
        arr2 = (np.arange(10) * np.arange(5)[np.newaxis].T)[::-1]
        ds = self.element((xs, ys, arr, arr2), kdims=['x', 'y'], vdims=['z', 'z2'], datatype=[self.datatype, 'dictionary'])
        sliced = self.element((xs[[1, 2, 3]], ys[[0, 1, 2]], arr[[0, 1, 2], [1, 2, 3]], arr2[[0, 1, 2], [1, 2, 3]]), kdims=['x', 'y'], vdims=['z', 'z2'], datatype=['dictionary'])
        self.assertEqual(ds.ndloc[[0, 1, 2], [1, 2, 3]], sliced)

    def test_reindex_drop_scalars_xs(self):
        reindexed = self.dataset_grid.ndloc[:, 0].reindex()
        ds = Dataset((self.grid_ys, self.grid_zs[:, 0]), 'y', 'z')
        self.assertEqual(reindexed, ds)

    def test_reindex_drop_scalars_ys(self):
        reindexed = self.dataset_grid.ndloc[0].reindex()
        ds = Dataset((self.grid_xs, self.grid_zs[0]), 'x', 'z')
        self.assertEqual(reindexed, ds)

    def test_reindex_2d_grid_to_1d(self):
        with DatatypeContext([self.datatype, 'dictionary', 'dataframe'], self.dataset_grid):
            ds = self.dataset_grid.reindex(kdims=['x'])
        with DatatypeContext([self.datatype, 'dictionary', 'dataframe'], Dataset):
            self.assertEqual(ds, Dataset(self.dataset_grid.columns(), 'x', 'z'))

    def test_mask_2d_array(self):
        array = np.random.rand(4, 3)
        ds = Dataset(([0, 1, 2], [1, 2, 3, 4], array), ['x', 'y'], 'z')
        mask = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 0, 1]], dtype='bool')
        masked = ds.clone(ds.interface.mask(ds, mask))
        masked_array = masked.dimension_values(2, flat=False)
        expected = array.copy()
        expected[mask] = np.nan
        self.assertEqual(masked_array, expected)

    def test_mask_2d_array_x_reversed(self):
        array = np.random.rand(4, 3)
        ds = Dataset(([0, 1, 2][::-1], [1, 2, 3, 4], array[:, ::-1]), ['x', 'y'], 'z')
        mask = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 0, 1]], dtype='bool')
        masked = ds.clone(ds.interface.mask(ds, mask))
        masked_array = masked.dimension_values(2, flat=False)
        expected = array.copy()
        expected[mask] = np.nan
        self.assertEqual(masked_array, expected)

    def test_mask_2d_array_y_reversed(self):
        array = np.random.rand(4, 3)
        ds = Dataset(([0, 1, 2], [1, 2, 3, 4][::-1], array[::-1]), ['x', 'y'], 'z')
        mask = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 0, 1]], dtype='bool')
        masked = ds.clone(ds.interface.mask(ds, mask))
        masked_array = masked.dimension_values(2, flat=False)
        expected = array.copy()
        expected[mask] = np.nan
        self.assertEqual(masked_array, expected)

    def test_mask_2d_array_xy_reversed(self):
        array = np.random.rand(4, 3)
        ds = Dataset(([0, 1, 2][::-1], [1, 2, 3, 4][::-1], array[::-1, ::-1]), ['x', 'y'], 'z')
        mask = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 0, 1]], dtype='bool')
        masked = ds.clone(ds.interface.mask(ds, mask))
        masked_array = masked.dimension_values(2, flat=False)
        expected = array.copy()
        expected[mask] = np.nan
        self.assertEqual(masked_array, expected)

    def test_dataset_transform_replace_kdim_on_grid(self):
        transformed = self.dataset_grid.transform(x=dim('x') * 2)
        expected = self.element(([0, 2], self.grid_ys, self.grid_zs), ['x', 'y'], ['z'])
        self.assertEqual(transformed, expected)

    def test_dataset_transform_replace_vdim_on_grid(self):
        transformed = self.dataset_grid.transform(z=dim('z') * 2)
        expected = self.element((self.grid_xs, self.grid_ys, self.grid_zs * 2), ['x', 'y'], ['z'])
        self.assertEqual(transformed, expected)

    def test_dataset_transform_replace_kdim_on_inverted_grid(self):
        transformed = self.dataset_grid_inv.transform(x=dim('x') * 2)
        expected = self.element(([2, 0], self.grid_ys[::-1], self.grid_zs), ['x', 'y'], ['z'])
        self.assertEqual(transformed, expected)

    def test_dataset_transform_replace_vdim_on_inverted_grid(self):
        transformed = self.dataset_grid_inv.transform(z=dim('z') * 2)
        expected = self.element((self.grid_xs[::-1], self.grid_ys[::-1], self.grid_zs * 2), ['x', 'y'], ['z'])
        self.assertEqual(transformed, expected)