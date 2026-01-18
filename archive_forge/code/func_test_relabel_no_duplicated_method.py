import pandas as pd
import pandas._testing as tm
def test_relabel_no_duplicated_method():
    df = pd.DataFrame({'A': [1, 2, 1, 2], 'B': [1, 2, 3, 4]})
    result = df['A'].agg(foo='sum')
    expected = df['A'].agg({'foo': 'sum'})
    tm.assert_series_equal(result, expected)
    result = df['B'].agg(foo='min', bar='max')
    expected = df['B'].agg({'foo': 'min', 'bar': 'max'})
    tm.assert_series_equal(result, expected)
    msg = 'using Series.[sum|min|max]'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df['B'].agg(foo=sum, bar=min, cat='max')
    msg = 'using Series.[sum|min|max]'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        expected = df['B'].agg({'foo': sum, 'bar': min, 'cat': 'max'})
    tm.assert_series_equal(result, expected)