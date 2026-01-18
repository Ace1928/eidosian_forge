import pytest
from pandas import (
from pandas.tests.io.pytables.common import (
def test_retain_index_attributes(setup_path, unit):
    dti = date_range('2000-1-1', periods=3, freq='h', unit=unit)
    df = DataFrame({'A': Series(range(3), index=dti)})
    with ensure_clean_store(setup_path) as store:
        _maybe_remove(store, 'data')
        store.put('data', df, format='table')
        result = store.get('data')
        tm.assert_frame_equal(df, result)
        for attr in ['freq', 'tz', 'name']:
            for idx in ['index', 'columns']:
                assert getattr(getattr(df, idx), attr, None) == getattr(getattr(result, idx), attr, None)
        dti2 = date_range('2002-1-1', periods=3, freq='D', unit=unit)
        with tm.assert_produces_warning(errors.AttributeConflictWarning):
            df2 = DataFrame({'A': Series(range(3), index=dti2)})
            store.append('data', df2)
        assert store.get_storer('data').info['index']['freq'] is None
        _maybe_remove(store, 'df2')
        dti3 = DatetimeIndex(['2001-01-01', '2001-01-02', '2002-01-01'], dtype=f'M8[{unit}]')
        df2 = DataFrame({'A': Series(range(3), index=dti3)})
        store.append('df2', df2)
        dti4 = date_range('2002-1-1', periods=3, freq='D', unit=unit)
        df3 = DataFrame({'A': Series(range(3), index=dti4)})
        store.append('df2', df3)