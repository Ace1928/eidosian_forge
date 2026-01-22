import os
import re
import numpy as np
import pandas
import pyarrow
import pytest
from pandas._testing import ensure_clean
from pandas.core.dtypes.common import is_list_like
from pyhdk import __version__ as hdk_version
from modin.config import StorageFormat
from modin.tests.interchange.dataframe_protocol.hdk.utils import split_df_into_chunks
from modin.tests.pandas.utils import (
from .utils import ForceHdkImport, eval_io, run_and_compare, set_execution_mode
import modin.pandas as pd
from modin.experimental.core.execution.native.implementations.hdk_on_native.calcite_serializer import (
from modin.experimental.core.execution.native.implementations.hdk_on_native.df_algebra import (
from modin.experimental.core.execution.native.implementations.hdk_on_native.partitioning.partition_manager import (
from modin.pandas.io import from_arrow
from modin.tests.pandas.utils import (
from modin.utils import try_cast_to_pandas
class TestBadData:
    bad_for_arrow = {'a': ['a', [[1, 2], [3]], [3, 4]], 'b': ['b', [1, 2], [3, 4]], 'c': ['1', '2', 3]}
    bad_for_hdk = {'b': [[1, 2], [3, 4], [5, 6]], 'c': ['1', '2', '3']}
    ok_data = {'d': np.arange(3), 'e': np.arange(3), 'f': np.arange(3)}

    def _get_pyarrow_table(self, obj):
        if not isinstance(obj, (pandas.DataFrame, pandas.Series)):
            obj = pandas.DataFrame(obj)
        return pyarrow.Table.from_pandas(obj)

    @pytest.mark.parametrize('data', [bad_for_arrow, bad_for_hdk])
    def test_construct(self, data):

        def applier(df, *args, **kwargs):
            return repr(df)
        run_and_compare(applier, data=data, force_lazy=False)

    def test_from_arrow(self):
        at = self._get_pyarrow_table(self.bad_for_hdk)
        pd_df = pandas.DataFrame(self.bad_for_hdk)
        md_df = pd.utils.from_arrow(at)
        repr(md_df)
        df_equals(md_df, pd_df)

    @pytest.mark.parametrize('data', [bad_for_arrow, bad_for_hdk])
    def test_methods(self, data):

        def applier(df, *args, **kwargs):
            return df.T.drop(columns=[0])
        run_and_compare(applier, data=data, force_lazy=False)

    def test_with_normal_frame(self):

        def applier(df1, df2, *args, **kwargs):
            return df2.join(df1)
        run_and_compare(applier, data=self.bad_for_hdk, data2=self.ok_data, force_lazy=False)

    def test_heterogenous_fillna(self):

        def fillna(df, **kwargs):
            return df['d'].fillna('a')
        run_and_compare(fillna, data=self.ok_data, force_lazy=False)

    @pytest.mark.parametrize('md_df_constructor', [pytest.param(pd.DataFrame, id='from_pandas_dataframe'), pytest.param(lambda pd_df: from_arrow(pyarrow.Table.from_pandas(pd_df)), id='from_pyarrow_table')])
    def test_uint(self, md_df_constructor):
        """
        Verify that unsigned integer data could be imported-exported via HDK with no errors.

        Originally, HDK does not support unsigned integers, there's a logic in Modin that
        upcasts unsigned types to the compatible ones prior importing to HDK.
        """
        pd_df = pandas.DataFrame({'uint8_in_int_bounds': np.array([1, 2, 3], dtype='uint8'), 'uint8_out-of_int_bounds': np.array([2 ** 8 - 1, 2 ** 8 - 2, 2 ** 8 - 3], dtype='uint8'), 'uint16_in_int_bounds': np.array([1, 2, 3], dtype='uint16'), 'uint16_out-of_int_bounds': np.array([2 ** 16 - 1, 2 ** 16 - 2, 2 ** 16 - 3], dtype='uint16'), 'uint32_in_int_bounds': np.array([1, 2, 3], dtype='uint32'), 'uint32_out-of_int_bounds': np.array([2 ** 32 - 1, 2 ** 32 - 2, 2 ** 32 - 3], dtype='uint32'), 'uint64_in_int_bounds': np.array([1, 2, 3], dtype='uint64')})
        md_df = md_df_constructor(pd_df)
        with ForceHdkImport(md_df) as instance:
            md_df_exported = instance.export_frames()[0]
            result = md_df_exported.values
            reference = pd_df.values
            np.testing.assert_array_equal(result, reference)

    @pytest.mark.parametrize('md_df_constructor', [pytest.param(pd.DataFrame, id='from_pandas_dataframe'), pytest.param(lambda pd_df: from_arrow(pyarrow.Table.from_pandas(pd_df)), id='from_pyarrow_table')])
    def test_uint_overflow(self, md_df_constructor):
        """
        Verify that the exception is arisen when overflow occurs due to 'uint -> int' compatibility conversion.

        Originally, HDK does not support unsigned integers, there's a logic in Modin that upcasts
        unsigned types to the compatible ones prior importing to HDK. This test ensures that the
        error is arisen when such conversion causes a data loss.
        """
        md_df = md_df_constructor(pandas.DataFrame({'col': np.array([2 ** 64 - 1, 2 ** 64 - 2, 2 ** 64 - 3], dtype='uint64')}))
        with pytest.raises(OverflowError):
            with ForceHdkImport(md_df):
                pass

    def test_uint_serialization(self):
        df = pd.DataFrame({'A': [np.nan, 1]})
        assert df.fillna(np.uint8(np.iinfo(np.uint8).max)).sum()[0] == np.iinfo(np.uint8).max + 1
        assert df.fillna(np.uint16(np.iinfo(np.uint16).max)).sum()[0] == np.iinfo(np.uint16).max + 1
        assert df.fillna(np.uint32(np.iinfo(np.uint32).max)).sum()[0] == np.iinfo(np.uint32).max + 1
        assert df.fillna(np.uint64(np.iinfo(np.int64).max - 1)).sum()[0] == np.iinfo(np.int64).max
        df = pd.DataFrame({'A': [np.iinfo(np.uint8).max, 1]})
        assert df.astype(np.uint8).sum()[0] == np.iinfo(np.uint8).max + 1
        df = pd.DataFrame({'A': [np.iinfo(np.uint16).max, 1]})
        assert df.astype(np.uint16).sum()[0] == np.iinfo(np.uint16).max + 1
        df = pd.DataFrame({'A': [np.iinfo(np.uint32).max, 1]})
        assert df.astype(np.uint32).sum()[0] == np.iinfo(np.uint32).max + 1
        df = pd.DataFrame({'A': [np.iinfo(np.int64).max - 1, 1]})
        assert df.astype(np.uint64).sum()[0] == np.iinfo(np.int64).max

    def test_mean_sum(self):
        all_codes = np.typecodes['All']
        exclude_codes = np.typecodes['Datetime'] + np.typecodes['Complex'] + 'gSUVO'
        supported_codes = set(all_codes) - set(exclude_codes)

        def test(df, dtype_code, operation, **kwargs):
            df = type(df)({'A': [0, 1], 'B': [1, 0]}, dtype=np.dtype(dtype_code))
            return getattr(df, operation)()
        for c in supported_codes:
            for op in ('sum', 'mean'):
                run_and_compare(test, data={}, dtype_code=c, operation=op)