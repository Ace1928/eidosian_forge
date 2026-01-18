import pandas as pd
import pandas._testing as tm
def test_relabel_duplicated_method():
    df = pd.DataFrame({'A': [1, 2, 1, 2], 'B': [1, 2, 3, 4]})
    result = df['A'].agg(foo='sum', bar='sum')
    expected = pd.Series([6, 6], index=['foo', 'bar'], name='A')
    tm.assert_series_equal(result, expected)
    msg = 'using Series.min'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df['B'].agg(foo=min, bar='min')
    expected = pd.Series([1, 1], index=['foo', 'bar'], name='B')
    tm.assert_series_equal(result, expected)