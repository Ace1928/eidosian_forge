import datetime
import numpy as np
import pandas as pd
from holoviews import Dataset, Dimension, HoloMap
from holoviews.core.data import concat
from holoviews.core.data.interface import DataError
from holoviews.element import Curve, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
class HeterogeneousColumnTests(HomogeneousColumnTests):
    """
    Tests for data formats that allow dataset to have varied types
    """
    __test__ = False

    def init_column_data(self):
        self.kdims = ['Gender', 'Age']
        self.vdims = ['Weight', 'Height']
        self.gender, self.age = (np.array(['M', 'M', 'F']), np.array([10, 16, 12]))
        self.weight, self.height = (np.array([15, 18, 10]), np.array([0.8, 0.6, 0.8]))
        self.table = Dataset({'Gender': self.gender, 'Age': self.age, 'Weight': self.weight, 'Height': self.height}, kdims=self.kdims, vdims=self.vdims)
        self.alias_kdims = [('gender', 'Gender'), ('age', 'Age')]
        self.alias_vdims = [('weight', 'Weight'), ('height', 'Height')]
        self.alias_table = Dataset({'gender': self.gender, 'age': self.age, 'weight': self.weight, 'height': self.height}, kdims=self.alias_kdims, vdims=self.alias_vdims)
        super().init_column_data()
        self.ys = np.linspace(0, 1, 11)
        self.zs = np.sin(self.xs)
        self.dataset_ht = Dataset({'x': self.xs, 'y': self.ys}, kdims=['x'], vdims=['y'])

    def test_dataset_dataframe_init_ht(self):
        """Tests support for heterogeneous DataFrames"""
        dataset = Dataset(pd.DataFrame({'x': self.xs, 'y': self.ys}), kdims=['x'], vdims=['y'])
        self.assertTrue(isinstance(dataset.data, self.data_type))

    def test_dataset_dataframe_init_ht_alias(self):
        """Tests support for heterogeneous DataFrames"""
        dataset = Dataset(pd.DataFrame({'x': self.xs, 'y': self.ys}), kdims=[('x', 'X')], vdims=[('y', 'Y')])
        self.assertTrue(isinstance(dataset.data, self.data_type))

    def test_dataset_dataset_ht_dtypes(self):
        ds = self.table
        self.assertEqual(ds.interface.dtype(ds, 'Gender'), np.dtype('object'))
        self.assertEqual(ds.interface.dtype(ds, 'Age'), np.dtype(int))
        self.assertEqual(ds.interface.dtype(ds, 'Weight'), np.dtype(int))
        self.assertEqual(ds.interface.dtype(ds, 'Height'), np.dtype('float64'))

    def test_dataset_expanded_dimvals_ht(self):
        data = self.table.dimension_values('Gender', expanded=False)
        self.assertEqual(np.sort(data), np.array(['F', 'M']))

    def test_dataset_implicit_indexing_init(self):
        dataset = Scatter(self.ys, kdims=['x'], vdims=['y'])
        self.assertTrue(isinstance(dataset.data, self.data_type))

    def test_dataset_tuple_init(self):
        dataset = Dataset((self.xs, self.ys), kdims=['x'], vdims=['y'])
        self.assertTrue(isinstance(dataset.data, self.data_type))

    def test_dataset_tuple_init_alias(self):
        dataset = Dataset((self.xs, self.ys), kdims=[('x', 'X')], vdims=[('y', 'Y')])
        self.assertTrue(isinstance(dataset.data, self.data_type))

    def test_dataset_simple_zip_init(self):
        dataset = Dataset(zip(self.xs, self.ys), kdims=['x'], vdims=['y'])
        self.assertTrue(isinstance(dataset.data, self.data_type))

    def test_dataset_simple_zip_init_alias(self):
        dataset = Dataset(zip(self.xs, self.ys), kdims=[('x', 'X')], vdims=[('y', 'Y')])
        self.assertTrue(isinstance(dataset.data, self.data_type))

    def test_dataset_zip_init(self):
        dataset = Dataset(zip(self.gender, self.age, self.weight, self.height), kdims=self.kdims, vdims=self.vdims)
        self.assertTrue(isinstance(dataset.data, self.data_type))

    def test_dataset_zip_init_alias(self):
        dataset = self.alias_table.clone(zip(self.gender, self.age, self.weight, self.height))
        self.assertTrue(isinstance(dataset.data, self.data_type))

    def test_dataset_odict_init(self):
        dataset = Dataset(dict(zip(self.xs, self.ys)), kdims=['A'], vdims=['B'])
        self.assertTrue(isinstance(dataset.data, self.data_type))

    def test_dataset_odict_init_alias(self):
        dataset = Dataset(dict(zip(self.xs, self.ys)), kdims=[('a', 'A')], vdims=[('b', 'B')])
        self.assertTrue(isinstance(dataset.data, self.data_type))

    def test_dataset_dict_init(self):
        dataset = Dataset(dict(zip(self.xs, self.ys)), kdims=['A'], vdims=['B'])
        self.assertTrue(isinstance(dataset.data, self.data_type))

    def test_dataset_range_with_dimension_range(self):
        dt64 = np.array([np.datetime64(datetime.datetime(2017, 1, i)) for i in range(1, 4)])
        ds = Dataset(dt64, [Dimension('Date', range=(dt64[0], dt64[-1]))])
        self.assertEqual(ds.range('Date'), (dt64[0], dt64[-1]))

    def test_dataset_redim_with_alias_dframe(self):
        test_df = pd.DataFrame({'x': range(10), 'y': range(0, 20, 2)})
        dataset = Dataset(test_df, kdims=[('x', 'X-label')], vdims=['y'])
        redim_df = pd.DataFrame({'X': range(10), 'y': range(0, 20, 2)})
        dataset_redim = Dataset(redim_df, kdims=['X'], vdims=['y'])
        self.assertEqual(dataset.redim(**{'X-label': 'X'}), dataset_redim)
        self.assertEqual(dataset.redim(x='X'), dataset_redim)

    def test_dataset_mixed_type_range(self):
        ds = Dataset((['A', 'B', 'C', None],), 'A')
        self.assertEqual(ds.range(0), ('A', 'C'))

    def test_dataset_nodata_range(self):
        table = self.table.clone(vdims=[Dimension('Weight', nodata=10), 'Height'])
        self.assertEqual(table.range('Weight'), (15, 18))

    def test_dataset_sort_vdim_ht(self):
        dataset = Dataset({'x': self.xs, 'y': -self.ys}, kdims=['x'], vdims=['y'])
        dataset_sorted = Dataset({'x': self.xs[::-1], 'y': -self.ys[::-1]}, kdims=['x'], vdims=['y'])
        self.assertEqual(dataset.sort('y'), dataset_sorted)

    def test_dataset_sort_string_ht(self):
        dataset_sorted = Dataset({'Gender': ['F', 'M', 'M'], 'Age': [12, 10, 16], 'Weight': [10, 15, 18], 'Height': [0.8, 0.8, 0.6]}, kdims=self.kdims, vdims=self.vdims)
        self.assertEqual(self.table.sort(), dataset_sorted)

    def test_dataset_sample_ht(self):
        samples = self.dataset_ht.sample([0, 5, 10]).dimension_values('y')
        self.assertEqual(samples, np.array([0, 0.5, 1]))

    def test_dataset_reduce_ht(self):
        reduced = Dataset({'Age': self.age, 'Weight': self.weight, 'Height': self.height}, kdims=self.kdims[1:], vdims=self.vdims)
        self.assertEqual(self.table.reduce(['Gender'], np.mean), reduced)

    def test_dataset_1D_reduce_ht(self):
        self.assertEqual(self.dataset_ht.reduce('x', np.mean), np.float64(0.5))

    def test_dataset_2D_reduce_ht(self):
        reduced = Dataset({'Weight': [14.333333333333334], 'Height': [0.7333333333333334]}, kdims=[], vdims=self.vdims)
        self.assertEqual(self.table.reduce(function=np.mean), reduced)

    def test_dataset_2D_partial_reduce_ht(self):
        dataset = Dataset({'x': self.xs, 'y': self.ys, 'z': self.zs}, kdims=['x', 'y'], vdims=['z'])
        reduced = Dataset({'x': self.xs, 'z': self.zs}, kdims=['x'], vdims=['z'])
        self.assertEqual(dataset.reduce(['y'], np.mean), reduced)

    def test_dataset_2D_aggregate_spread_fn_with_duplicates(self):
        dataset = Dataset({'x': np.array([0, 0, 1, 1]), 'y': np.array([0, 1, 2, 3]), 'z': np.array([1, 2, 3, 4])}, kdims=['x', 'y'], vdims=['z'])
        agg = dataset.aggregate('x', function=np.mean, spreadfn=np.var)
        self.assertEqual(agg, Dataset({'x': np.array([0, 1]), 'z': np.array([1.5, 3.5]), 'z_var': np.array([0.25, 0.25])}, kdims=['x'], vdims=['z', 'z_var']))

    def test_dataset_aggregate_ht(self):
        aggregated = Dataset({'Gender': ['M', 'F'], 'Weight': [16.5, 10], 'Height': [0.7, 0.8]}, kdims=self.kdims[:1], vdims=self.vdims)
        self.compare_dataset(self.table.aggregate(['Gender'], np.mean), aggregated)

    def test_dataset_aggregate_string_types(self):
        ds = Dataset({'Gender': ['M', 'M'], 'Weight': [20, 10], 'Name': ['Peter', 'Matt']}, kdims='Gender', vdims=['Weight', 'Name'])
        aggregated = Dataset({'Gender': ['M'], 'Weight': [15]}, kdims='Gender', vdims=['Weight'])
        self.compare_dataset(ds.aggregate(['Gender'], np.mean), aggregated)

    def test_dataset_aggregate_string_types_size(self):
        ds = Dataset({'Gender': ['M', 'M'], 'Weight': [20, 10], 'Name': ['Peter', 'Matt']}, kdims='Gender', vdims=['Weight', 'Name'])
        aggregated = Dataset({'Gender': ['M'], 'Weight': [2], 'Name': [2]}, kdims='Gender', vdims=['Weight', 'Name'])
        self.compare_dataset(ds.aggregate(['Gender'], np.size), aggregated)

    def test_dataset_aggregate_ht_alias(self):
        aggregated = Dataset({'gender': ['M', 'F'], 'weight': [16.5, 10], 'height': [0.7, 0.8]}, kdims=self.alias_kdims[:1], vdims=self.alias_vdims)
        self.compare_dataset(self.alias_table.aggregate('Gender', np.mean), aggregated)

    def test_dataset_2D_aggregate_partial_ht(self):
        dataset = Dataset({'x': self.xs, 'y': self.ys, 'z': self.zs}, kdims=['x', 'y'], vdims=['z'])
        reduced = Dataset({'x': self.xs, 'z': self.zs}, kdims=['x'], vdims=['z'])
        self.assertEqual(dataset.aggregate(['x'], np.mean), reduced)

    def test_dataset_empty_aggregate(self):
        dataset = Dataset([], kdims=self.kdims, vdims=self.vdims)
        aggregated = Dataset([], kdims=self.kdims[:1], vdims=self.vdims)
        self.compare_dataset(dataset.aggregate(['Gender'], np.mean), aggregated)

    def test_dataset_empty_aggregate_with_spreadfn(self):
        dataset = Dataset([], kdims=self.kdims, vdims=self.vdims)
        aggregated = Dataset([], kdims=self.kdims[:1], vdims=[d for vd in self.vdims for d in [vd, vd + '_std']])
        self.compare_dataset(dataset.aggregate(['Gender'], np.mean, np.std), aggregated)

    def test_dataset_groupby(self):
        group1 = {'Age': [10, 16], 'Weight': [15, 18], 'Height': [0.8, 0.6]}
        group2 = {'Age': [12], 'Weight': [10], 'Height': [0.8]}
        grouped = HoloMap([('M', Dataset(group1, kdims=['Age'], vdims=self.vdims)), ('F', Dataset(group2, kdims=['Age'], vdims=self.vdims))], kdims=['Gender'], sort=False)
        self.assertEqual(self.table.groupby(['Gender']), grouped)

    def test_dataset_groupby_alias(self):
        group1 = {'age': [10, 16], 'weight': [15, 18], 'height': [0.8, 0.6]}
        group2 = {'age': [12], 'weight': [10], 'height': [0.8]}
        grouped = HoloMap([('M', Dataset(group1, kdims=[('age', 'Age')], vdims=self.alias_vdims)), ('F', Dataset(group2, kdims=[('age', 'Age')], vdims=self.alias_vdims))], kdims=[('gender', 'Gender')], sort=False)
        self.assertEqual(self.alias_table.groupby('Gender'), grouped)

    def test_dataset_groupby_second_dim(self):
        group1 = {'Gender': ['M'], 'Weight': [15], 'Height': [0.8]}
        group2 = {'Gender': ['M'], 'Weight': [18], 'Height': [0.6]}
        group3 = {'Gender': ['F'], 'Weight': [10], 'Height': [0.8]}
        grouped = HoloMap([(10, Dataset(group1, kdims=['Gender'], vdims=self.vdims)), (16, Dataset(group2, kdims=['Gender'], vdims=self.vdims)), (12, Dataset(group3, kdims=['Gender'], vdims=self.vdims))], kdims=['Age'], sort=False)
        self.assertEqual(self.table.groupby(['Age']), grouped)

    def test_dataset_groupby_dynamic(self):
        grouped_dataset = self.table.groupby('Gender', dynamic=True)
        self.assertEqual(grouped_dataset['M'], self.table.select(Gender='M').reindex(['Age']))
        self.assertEqual(grouped_dataset['F'], self.table.select(Gender='F').reindex(['Age']))

    def test_dataset_groupby_dynamic_alias(self):
        grouped_dataset = self.alias_table.groupby('Gender', dynamic=True)
        self.assertEqual(grouped_dataset['M'], self.alias_table.select(gender='M').reindex(['Age']))
        self.assertEqual(grouped_dataset['F'], self.alias_table.select(gender='F').reindex(['Age']))

    def test_dataset_add_dimensions_value_ht(self):
        table = self.dataset_ht.add_dimension('z', 1, 0)
        self.assertEqual(table.kdims[1], 'z')
        self.compare_arrays(table.dimension_values('z'), np.zeros(table.shape[0]))

    def test_dataset_add_dimensions_value_ht_alias(self):
        table = self.dataset_ht.add_dimension(('z', 'Z'), 1, 0)
        self.assertEqual(table.kdims[1], 'z')
        self.compare_arrays(table.dimension_values('z'), np.zeros(table.shape[0]))

    def test_dataset_add_dimensions_values_ht(self):
        table = self.dataset_ht.add_dimension('z', 1, range(1, 12))
        self.assertEqual(table.kdims[1], 'z')
        self.compare_arrays(table.dimension_values('z'), np.array(list(range(1, 12))))

    def test_redim_with_extra_dimension(self):
        dataset = self.dataset_ht.add_dimension('Temp', 0, 0).clone(kdims=['x', 'y'], vdims=[])
        redimmed = dataset.redim(x='Time')
        self.assertEqual(redimmed.dimension_values('Time'), self.dataset_ht.dimension_values('x'))

    def test_dataset_index_row_gender_female(self):
        indexed = Dataset({'Gender': ['F'], 'Age': [12], 'Weight': [10], 'Height': [0.8]}, kdims=self.kdims, vdims=self.vdims)
        row = self.table['F', :]
        self.assertEqual(row, indexed)

    def test_dataset_index_rows_gender_male(self):
        row = self.table['M', :]
        indexed = Dataset({'Gender': ['M', 'M'], 'Age': [10, 16], 'Weight': [15, 18], 'Height': [0.8, 0.6]}, kdims=self.kdims, vdims=self.vdims)
        self.assertEqual(row, indexed)

    def test_dataset_select_rows_gender_male(self):
        row = self.table.select(Gender='M')
        indexed = Dataset({'Gender': ['M', 'M'], 'Age': [10, 16], 'Weight': [15, 18], 'Height': [0.8, 0.6]}, kdims=self.kdims, vdims=self.vdims)
        self.assertEqual(row, indexed)

    def test_dataset_select_rows_gender_male_expr(self):
        row = self.table.select(selection_expr=dim('Gender') == 'M')
        indexed = Dataset({'Gender': ['M', 'M'], 'Age': [10, 16], 'Weight': [15, 18], 'Height': [0.8, 0.6]}, kdims=self.kdims, vdims=self.vdims)
        self.assertEqual(row, indexed)

    def test_dataset_select_rows_gender_male_alias(self):
        row = self.alias_table.select(Gender='M')
        alias_row = self.alias_table.select(gender='M')
        indexed = Dataset({'gender': ['M', 'M'], 'age': [10, 16], 'weight': [15, 18], 'height': [0.8, 0.6]}, kdims=self.alias_kdims, vdims=self.alias_vdims)
        self.assertEqual(row, indexed)
        self.assertEqual(alias_row, indexed)

    def test_dataset_index_row_age(self):
        indexed = Dataset({'Gender': ['F'], 'Age': [12], 'Weight': [10], 'Height': [0.8]}, kdims=self.kdims, vdims=self.vdims)
        self.assertEqual(self.table[:, 12], indexed)

    def test_dataset_index_item_table(self):
        indexed = Dataset({'Gender': ['F'], 'Age': [12], 'Weight': [10], 'Height': [0.8]}, kdims=self.kdims, vdims=self.vdims)
        self.assertEqual(self.table['F', 12], indexed)

    def test_dataset_index_value1(self):
        self.assertEqual(self.table['F', 12, 'Weight'], 10)

    def test_dataset_index_value2(self):
        self.assertEqual(self.table['F', 12, 'Height'], 0.8)

    def test_dataset_index_column_ht(self):
        self.compare_arrays(self.dataset_ht['y'], self.ys)

    def test_dataset_boolean_index(self):
        row = self.table[np.array([True, True, False])]
        indexed = Dataset({'Gender': ['M', 'M'], 'Age': [10, 16], 'Weight': [15, 18], 'Height': [0.8, 0.6]}, kdims=self.kdims, vdims=self.vdims)
        self.assertEqual(row, indexed)

    def test_dataset_value_dim_index(self):
        row = self.table[:, :, 'Weight']
        indexed = Dataset({'Gender': ['M', 'M', 'F'], 'Age': [10, 16, 12], 'Weight': [15, 18, 10]}, kdims=self.kdims, vdims=self.vdims[:1])
        self.assertEqual(row, indexed)

    def test_dataset_value_dim_scalar_index(self):
        row = self.table['M', 10, 'Weight']
        self.assertEqual(row, 15)

    def test_dataset_iloc_slice_rows(self):
        sliced = self.table.iloc[1:2]
        table = Dataset({'Gender': self.gender[1:2], 'Age': self.age[1:2], 'Weight': self.weight[1:2], 'Height': self.height[1:2]}, kdims=self.kdims, vdims=self.vdims)
        self.assertEqual(sliced, table)

    def test_dataset_iloc_slice_rows_slice_cols(self):
        sliced = self.table.iloc[1:2, 1:3]
        table = Dataset({'Age': self.age[1:2], 'Weight': self.weight[1:2]}, kdims=self.kdims[1:], vdims=self.vdims[:1])
        self.assertEqual(sliced, table)

    def test_dataset_iloc_slice_rows_list_cols(self):
        sliced = self.table.iloc[1:2, [1, 3]]
        table = Dataset({'Age': self.age[1:2], 'Height': self.height[1:2]}, kdims=self.kdims[1:], vdims=self.vdims[1:])
        self.assertEqual(sliced, table)

    def test_dataset_iloc_slice_rows_index_cols(self):
        sliced = self.table.iloc[1:2, 2]
        table = Dataset({'Weight': self.weight[1:2]}, kdims=[], vdims=self.vdims[:1])
        self.assertEqual(sliced, table)

    def test_dataset_iloc_list_rows(self):
        sliced = self.table.iloc[[0, 2]]
        table = Dataset({'Gender': self.gender[[0, 2]], 'Age': self.age[[0, 2]], 'Weight': self.weight[[0, 2]], 'Height': self.height[[0, 2]]}, kdims=self.kdims, vdims=self.vdims)
        self.assertEqual(sliced, table)

    def test_dataset_iloc_list_rows_list_cols(self):
        sliced = self.table.iloc[[0, 2], [0, 2]]
        table = Dataset({'Gender': self.gender[[0, 2]], 'Weight': self.weight[[0, 2]]}, kdims=self.kdims[:1], vdims=self.vdims[:1])
        self.assertEqual(sliced, table)

    def test_dataset_iloc_list_rows_list_cols_by_name(self):
        sliced = self.table.iloc[[0, 2], ['Gender', 'Weight']]
        table = Dataset({'Gender': self.gender[[0, 2]], 'Weight': self.weight[[0, 2]]}, kdims=self.kdims[:1], vdims=self.vdims[:1])
        self.assertEqual(sliced, table)

    def test_dataset_iloc_list_rows_slice_cols(self):
        sliced = self.table.iloc[[0, 2], slice(1, 3)]
        table = Dataset({'Age': self.age[[0, 2]], 'Weight': self.weight[[0, 2]]}, kdims=self.kdims[1:], vdims=self.vdims[:1])
        self.assertEqual(sliced, table)

    def test_dataset_iloc_index_rows_index_cols(self):
        indexed = self.table.iloc[1, 1]
        self.assertEqual(indexed, self.age[1])

    def test_dataset_iloc_index_rows_slice_cols(self):
        indexed = self.table.iloc[1, 1:3]
        table = Dataset({'Age': self.age[[1]], 'Weight': self.weight[[1]]}, kdims=self.kdims[1:], vdims=self.vdims[:1])
        self.assertEqual(indexed, table)

    def test_dataset_iloc_list_cols(self):
        sliced = self.table.iloc[:, [0, 2]]
        table = Dataset({'Gender': self.gender, 'Weight': self.weight}, kdims=self.kdims[:1], vdims=self.vdims[:1])
        self.assertEqual(sliced, table)

    def test_dataset_iloc_list_cols_by_name(self):
        sliced = self.table.iloc[:, ['Gender', 'Weight']]
        table = Dataset({'Gender': self.gender, 'Weight': self.weight}, kdims=self.kdims[:1], vdims=self.vdims[:1])
        self.assertEqual(sliced, table)

    def test_dataset_iloc_ellipsis_list_cols(self):
        sliced = self.table.iloc[..., [0, 2]]
        table = Dataset({'Gender': self.gender, 'Weight': self.weight}, kdims=self.kdims[:1], vdims=self.vdims[:1])
        self.assertEqual(sliced, table)

    def test_dataset_iloc_ellipsis_list_cols_by_name(self):
        sliced = self.table.iloc[..., ['Gender', 'Weight']]
        table = Dataset({'Gender': self.gender, 'Weight': self.weight}, kdims=self.kdims[:1], vdims=self.vdims[:1])
        self.assertEqual(sliced, table)

    def test_dataset_array_ht(self):
        self.assertEqual(self.dataset_ht.array(), np.column_stack([self.xs, self.ys]))

    def test_dataset_transform_replace_ht(self):
        transformed = self.table.transform(Age=dim('Age') ** 2, Weight=dim('Weight') * 2, Height=dim('Height') / 2.0)
        expected = Dataset({'Gender': self.gender, 'Age': self.age ** 2, 'Weight': self.weight * 2, 'Height': self.height / 2.0}, kdims=self.kdims, vdims=self.vdims)
        self.assertEqual(transformed, expected)

    def test_dataset_transform_add_ht(self):
        transformed = self.table.transform(combined=dim('Age') * dim('Weight'))
        expected = Dataset({'Gender': self.gender, 'Age': self.age, 'Weight': self.weight, 'Height': self.height, 'combined': self.age * self.weight}, kdims=self.kdims, vdims=self.vdims + ['combined'])
        self.assertEqual(transformed, expected)