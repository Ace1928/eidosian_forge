import textwrap
from io import BytesIO
import pytest
from sklearn.datasets._arff_parser import (
@pytest.mark.parametrize('parser_func', [pytest.param(_liac_arff_parser, marks=pytest.mark.xfail), _pandas_arff_parser])
def test_pandas_arff_parser_strip_no_quotes(parser_func):
    """Check that we properly parse with no quotes characters."""
    pd = pytest.importorskip('pandas')
    arff_file = BytesIO(textwrap.dedent("\n            @relation 'toy'\n            @attribute 'cat_without_quote' {A, B, C}\n            @attribute 'str_without_quote' string\n            @attribute 'str_internal_quote' string\n            @attribute 'class' numeric\n            @data\n            A,some text,'internal' quote,0\n            ").encode('utf-8'))
    columns_info = {'cat_without_quote': {'data_type': 'nominal', 'name': 'cat_without_quote'}, 'str_without_quote': {'data_type': 'string', 'name': 'str_without_quote'}, 'str_internal_quote': {'data_type': 'string', 'name': 'str_internal_quote'}, 'class': {'data_type': 'numeric', 'name': 'class'}}
    feature_names = ['cat_without_quote', 'str_without_quote', 'str_internal_quote']
    target_names = ['class']
    expected_values = {'cat_without_quote': 'A', 'str_without_quote': 'some text', 'str_internal_quote': "'internal' quote", 'class': 0}
    _, _, frame, _ = parser_func(arff_file, output_arrays_type='pandas', openml_columns_info=columns_info, feature_names_to_select=feature_names, target_names_to_select=target_names)
    assert frame.columns.tolist() == feature_names + target_names
    pd.testing.assert_series_equal(frame.iloc[0], pd.Series(expected_values, name=0))