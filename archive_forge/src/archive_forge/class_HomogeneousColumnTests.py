import datetime
import numpy as np
import pandas as pd
from holoviews import Dataset, Dimension, HoloMap
from holoviews.core.data import concat
from holoviews.core.data.interface import DataError
from holoviews.element import Curve, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
class HomogeneousColumnTests:
    """
    Tests for data formats that require all dataset to have the same
    type (e.g. numpy arrays)
    """
    __test__ = False

    def init_column_data(self):
        self.xs = np.array(range(11))
        self.xs_2 = self.xs ** 2
        self.y_ints = self.xs * 2
        self.dataset_hm = Dataset((self.xs, self.y_ints), kdims=['x'], vdims=['y'])
        self.dataset_hm_alias = Dataset((self.xs, self.y_ints), kdims=[('x', 'X')], vdims=[('y', 'Y')])

    def test_dataset_array_init_hm(self):
        dataset = Dataset(np.column_stack([self.xs, self.xs_2]), kdims=['x'], vdims=['x2'])
        self.assertTrue(isinstance(dataset.data, self.data_type))

    def test_dataset_dtypes(self):
        self.assertEqual(self.dataset_hm.interface.dtype(self.dataset_hm, 'x'), np.dtype(int))
        self.assertEqual(self.dataset_hm.interface.dtype(self.dataset_hm, 'y'), np.dtype(int))

    def test_dataset_array_init_hm_tuple_dims(self):
        dataset = Dataset(np.column_stack([self.xs, self.xs_2]), kdims=[('x', 'X')], vdims=[('x2', 'X2')])
        self.assertTrue(isinstance(dataset.data, self.data_type))

    def test_dataset_dataframe_init_hm(self):
        """Tests support for homogeneous DataFrames"""
        dataset = Dataset(pd.DataFrame({'x': self.xs, 'x2': self.xs_2}), kdims=['x'], vdims=['x2'])
        self.assertTrue(isinstance(dataset.data, self.data_type))

    def test_dataset_dataframe_init_hm_alias(self):
        """Tests support for homogeneous DataFrames"""
        dataset = Dataset(pd.DataFrame({'x': self.xs, 'x2': self.xs_2}), kdims=[('x', 'X-label')], vdims=[('x2', 'X2-label')])
        self.assertTrue(isinstance(dataset.data, self.data_type))

    def test_dataset_empty_list_init(self):
        dataset = Dataset([], kdims=['x'], vdims=['y'])
        for d in 'xy':
            self.assertEqual(dataset.dimension_values(d), np.array([]))

    def test_dataset_dict_dim_not_found_raises_on_array(self):
        with self.assertRaises(ValueError):
            Dataset({'x': np.zeros(5)}, kdims=['Test'], vdims=[])

    def test_dataset_dict_dim_not_found_raises_on_scalar(self):
        with self.assertRaises(ValueError):
            Dataset({'x': 1}, kdims=['Test'], vdims=[])

    def test_dataset_shape(self):
        self.assertEqual(self.dataset_hm.shape, (11, 2))

    def test_dataset_range(self):
        self.assertEqual(self.dataset_hm.range('y'), (0, 20))

    def test_dataset_closest(self):
        closest = self.dataset_hm.closest([0.51, 1, 9.9])
        self.assertEqual(closest, [1.0, 1.0, 10.0])

    def test_dataset_sort_hm(self):
        ds = Dataset(([2, 2, 1], [2, 1, 2], [0.1, 0.2, 0.3]), kdims=['x', 'y'], vdims=['z']).sort()
        ds_sorted = Dataset(([1, 2, 2], [2, 1, 2], [0.3, 0.2, 0.1]), kdims=['x', 'y'], vdims=['z'])
        self.assertEqual(ds.sort(), ds_sorted)

    def test_dataset_sort_reverse_hm(self):
        ds = Dataset(([2, 1, 2, 1], [2, 2, 1, 1], [0.1, 0.2, 0.3, 0.4]), kdims=['x', 'y'], vdims=['z'])
        ds_sorted = Dataset(([2, 2, 1, 1], [2, 1, 2, 1], [0.1, 0.3, 0.2, 0.4]), kdims=['x', 'y'], vdims=['z'])
        self.assertEqual(ds.sort(reverse=True), ds_sorted)

    def test_dataset_sort_vdim_hm(self):
        xs_2 = np.array(self.xs_2)
        dataset = Dataset(np.column_stack([self.xs, -xs_2]), kdims=['x'], vdims=['y'])
        dataset_sorted = Dataset(np.column_stack([self.xs[::-1], -xs_2[::-1]]), kdims=['x'], vdims=['y'])
        self.assertEqual(dataset.sort('y'), dataset_sorted)

    def test_dataset_sort_reverse_vdim_hm(self):
        xs_2 = np.array(self.xs_2)
        dataset = Dataset(np.column_stack([self.xs, -xs_2]), kdims=['x'], vdims=['y'])
        dataset_sorted = Dataset(np.column_stack([self.xs, -xs_2]), kdims=['x'], vdims=['y'])
        self.assertEqual(dataset.sort('y', reverse=True), dataset_sorted)

    def test_dataset_sort_vdim_hm_alias(self):
        xs_2 = np.array(self.xs_2)
        dataset = Dataset(np.column_stack([self.xs, -xs_2]), kdims=[('x', 'X-label')], vdims=[('y', 'Y-label')])
        dataset_sorted = Dataset(np.column_stack([self.xs[::-1], -xs_2[::-1]]), kdims=[('x', 'X-label')], vdims=[('y', 'Y-label')])
        self.assertEqual(dataset.sort('y'), dataset_sorted)
        self.assertEqual(dataset.sort('Y-label'), dataset_sorted)

    def test_dataset_redim_hm_kdim(self):
        redimmed = self.dataset_hm.redim(x='Time')
        self.assertEqual(redimmed.dimension_values('Time'), self.dataset_hm.dimension_values('x'))

    def test_dataset_redim_hm_kdim_range_aux(self):
        redimmed = self.dataset_hm.redim.range(x=(-100, 3))
        self.assertEqual(redimmed.kdims[0].range, (-100, 3))

    def test_dataset_redim_hm_kdim_soft_range_aux(self):
        redimmed = self.dataset_hm.redim.soft_range(x=(-100, 30))
        self.assertEqual(redimmed.kdims[0].soft_range, (-100, 30))

    def test_dataset_redim_hm_kdim_alias(self):
        redimmed = self.dataset_hm_alias.redim(x='Time')
        self.assertEqual(redimmed.dimension_values('Time'), self.dataset_hm_alias.dimension_values('x'))

    def test_dataset_redim_hm_vdim(self):
        redimmed = self.dataset_hm.redim(y='Value')
        self.assertEqual(redimmed.dimension_values('Value'), self.dataset_hm.dimension_values('y'))

    def test_dataset_redim_hm_vdim_alias(self):
        redimmed = self.dataset_hm_alias.redim(y=Dimension(('val', 'Value')))
        self.assertEqual(redimmed.dimension_values('Value'), self.dataset_hm_alias.dimension_values('y'))

    def test_dataset_sample_hm(self):
        samples = self.dataset_hm.sample([0, 5, 10]).dimension_values('y')
        self.assertEqual(samples, np.array([0, 10, 20]))

    def test_dataset_sample_hm_alias(self):
        samples = self.dataset_hm_alias.sample([0, 5, 10]).dimension_values('y')
        self.assertEqual(samples, np.array([0, 10, 20]))

    def test_dataset_array_hm(self):
        self.assertEqual(self.dataset_hm.array(), np.column_stack([self.xs, self.y_ints]))

    def test_dataset_array_hm_alias(self):
        self.assertEqual(self.dataset_hm_alias.array(), np.column_stack([self.xs, self.y_ints]))

    def test_dataset_add_dimensions_value_hm(self):
        table = self.dataset_hm.add_dimension('z', 1, 0)
        self.assertEqual(table.kdims[1], 'z')
        self.compare_arrays(table.dimension_values('z'), np.zeros(table.shape[0]))

    def test_dataset_add_dimensions_values_hm(self):
        table = self.dataset_hm.add_dimension('z', 1, range(1, 12))
        self.assertEqual(table.kdims[1], 'z')
        self.compare_arrays(table.dimension_values('z'), np.array(list(range(1, 12))))

    def test_dataset_slice_hm(self):
        dataset_slice = Dataset({'x': range(5, 9), 'y': [2 * i for i in range(5, 9)]}, kdims=['x'], vdims=['y'])
        self.assertEqual(self.dataset_hm[5:9], dataset_slice)

    def test_dataset_slice_hm_alias(self):
        dataset_slice = Dataset({'x': range(5, 9), 'y': [2 * i for i in range(5, 9)]}, kdims=[('x', 'X')], vdims=[('y', 'Y')])
        self.assertEqual(self.dataset_hm_alias[5:9], dataset_slice)

    def test_dataset_slice_fn_hm(self):
        dataset_slice = Dataset({'x': range(5, 9), 'y': [2 * i for i in range(5, 9)]}, kdims=['x'], vdims=['y'])
        self.assertEqual(self.dataset_hm[lambda x: (x >= 5) & (x < 9)], dataset_slice)

    def test_dataset_1D_reduce_hm(self):
        dataset = Dataset({'x': self.xs, 'y': self.y_ints}, kdims=['x'], vdims=['y'])
        self.assertEqual(dataset.reduce('x', np.mean), 10)

    def test_dataset_1D_reduce_hm_alias(self):
        dataset = Dataset({'x': self.xs, 'y': self.y_ints}, kdims=[('x', 'X')], vdims=[('y', 'Y')])
        self.assertEqual(dataset.reduce('X', np.mean), 10)

    def test_dataset_2D_reduce_hm(self):
        dataset = Dataset({'x': self.xs, 'y': self.y_ints, 'z': [el ** 2 for el in self.y_ints]}, kdims=['x', 'y'], vdims=['z'])
        self.assertEqual(np.array(dataset.reduce(['x', 'y'], np.mean)), np.array(140))

    def test_dataset_2D_aggregate_partial_hm(self):
        z_ints = [el ** 2 for el in self.y_ints]
        dataset = Dataset({'x': self.xs, 'y': self.y_ints, 'z': z_ints}, kdims=['x', 'y'], vdims=['z'])
        self.assertEqual(dataset.aggregate(['x'], np.mean), Dataset({'x': self.xs, 'z': z_ints}, kdims=['x'], vdims=['z']))

    def test_dataset_index_column_idx_hm(self):
        self.assertEqual(self.dataset_hm[5], self.y_ints[5])

    def test_dataset_index_column_ht(self):
        self.compare_arrays(self.dataset_hm['y'], self.y_ints)

    def test_dataset_iloc_slice_rows(self):
        sliced = self.dataset_hm.iloc[1:4]
        table = Dataset({'x': self.xs[1:4], 'y': self.y_ints[1:4]}, kdims=['x'], vdims=['y'], datatype=['dictionary'])
        self.assertEqual(sliced, table)

    def test_dataset_iloc_slice_rows_slice_cols(self):
        sliced = self.dataset_hm.iloc[1:4, 1:]
        table = Dataset({'y': self.y_ints[1:4]}, kdims=[], vdims=['y'], datatype=['dictionary'])
        self.assertEqual(sliced, table)

    def test_dataset_iloc_slice_rows_list_cols(self):
        sliced = self.dataset_hm.iloc[1:4, [0, 1]]
        table = Dataset({'x': self.xs[1:4], 'y': self.y_ints[1:4]}, kdims=['x'], vdims=['y'], datatype=['dictionary'])
        self.assertEqual(sliced, table)

    def test_dataset_iloc_slice_rows_index_cols(self):
        sliced = self.dataset_hm.iloc[1:4, 1]
        table = Dataset({'y': self.y_ints[1:4]}, kdims=[], vdims=['y'], datatype=['dictionary'])
        self.assertEqual(sliced, table)

    def test_dataset_iloc_list_rows(self):
        sliced = self.dataset_hm.iloc[[0, 2]]
        table = Dataset({'x': self.xs[[0, 2]], 'y': self.y_ints[[0, 2]]}, kdims=['x'], vdims=['y'], datatype=['dictionary'])
        self.assertEqual(sliced, table)

    def test_dataset_iloc_list_rows_list_cols(self):
        sliced = self.dataset_hm.iloc[[0, 2], [0, 1]]
        table = Dataset({'x': self.xs[[0, 2]], 'y': self.y_ints[[0, 2]]}, kdims=['x'], vdims=['y'], datatype=['dictionary'])
        self.assertEqual(sliced, table)

    def test_dataset_iloc_list_rows_list_cols_by_name(self):
        sliced = self.dataset_hm.iloc[[0, 2], ['x', 'y']]
        table = Dataset({'x': self.xs[[0, 2]], 'y': self.y_ints[[0, 2]]}, kdims=['x'], vdims=['y'], datatype=['dictionary'])
        self.assertEqual(sliced, table)

    def test_dataset_iloc_list_rows_slice_cols(self):
        sliced = self.dataset_hm.iloc[[0, 2], slice(0, 2)]
        table = Dataset({'x': self.xs[[0, 2]], 'y': self.y_ints[[0, 2]]}, kdims=['x'], vdims=['y'], datatype=['dictionary'])
        self.assertEqual(sliced, table)

    def test_dataset_iloc_index_rows_index_cols(self):
        indexed = self.dataset_hm.iloc[1, 1]
        self.assertEqual(indexed, self.y_ints[1])

    def test_dataset_iloc_index_rows_slice_cols(self):
        indexed = self.dataset_hm.iloc[1, :2]
        table = Dataset({'x': self.xs[[1]], 'y': self.y_ints[[1]]}, kdims=['x'], vdims=['y'], datatype=['dictionary'])
        self.assertEqual(indexed, table)

    def test_dataset_iloc_list_cols(self):
        sliced = self.dataset_hm.iloc[:, [0, 1]]
        table = Dataset({'x': self.xs, 'y': self.y_ints}, kdims=['x'], vdims=['y'], datatype=['dictionary'])
        self.assertEqual(sliced, table)

    def test_dataset_iloc_list_cols_by_name(self):
        sliced = self.dataset_hm.iloc[:, ['x', 'y']]
        table = Dataset({'x': self.xs, 'y': self.y_ints}, kdims=['x'], vdims=['y'], datatype=['dictionary'])
        self.assertEqual(sliced, table)

    def test_dataset_iloc_ellipsis_list_cols(self):
        sliced = self.dataset_hm.iloc[..., [0, 1]]
        table = Dataset({'x': self.xs, 'y': self.y_ints}, kdims=['x'], vdims=['y'], datatype=['dictionary'])
        self.assertEqual(sliced, table)

    def test_dataset_iloc_ellipsis_list_cols_by_name(self):
        sliced = self.dataset_hm.iloc[..., ['x', 'y']]
        table = Dataset({'x': self.xs, 'y': self.y_ints}, kdims=['x'], vdims=['y'], datatype=['dictionary'])
        self.assertEqual(sliced, table)

    def test_dataset_get_array_by_dimension(self):
        arr = self.dataset_hm.array(['x'])
        self.assertEqual(arr, self.xs[:, np.newaxis])

    def test_dataset_get_dframe(self):
        df = self.dataset_hm.dframe()
        self.assertEqual(df.x.values, self.xs)
        self.assertEqual(df.y.values, self.y_ints)

    def test_dataset_get_dframe_by_dimension(self):
        df = self.dataset_hm.dframe(['x'])
        self.assertEqual(df, pd.DataFrame({'x': self.xs}, dtype=df.dtypes.iloc[0]))

    def test_dataset_transform_replace_hm(self):
        transformed = self.dataset_hm.transform(y=dim('y') * 2)
        expected = Dataset((self.xs, self.y_ints * 2), 'x', 'y')
        self.assertEqual(transformed, expected)

    def test_dataset_transform_add_hm(self):
        transformed = self.dataset_hm.transform(y2=dim('y') * 2)
        expected = Dataset((self.xs, self.y_ints, self.y_ints * 2), 'x', ['y', 'y2'])
        self.assertEqual(transformed, expected)