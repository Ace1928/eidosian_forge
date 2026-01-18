import modin.pandas as pd
from modin.pandas.api.extensions import register_series_accessor
def test_series_extension_simple_method():
    expected_string_val = 'Some string value'
    method_name = 'new_method'
    ser = pd.Series([1, 2, 3])

    @register_series_accessor(method_name)
    def my_method_implementation(self):
        return expected_string_val
    assert method_name in pd.series._SERIES_EXTENSIONS_.keys()
    assert pd.series._SERIES_EXTENSIONS_[method_name] is my_method_implementation
    assert ser.new_method() == expected_string_val