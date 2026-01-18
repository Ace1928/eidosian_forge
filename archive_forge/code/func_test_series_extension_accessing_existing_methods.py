import modin.pandas as pd
from modin.pandas.api.extensions import register_series_accessor
def test_series_extension_accessing_existing_methods():
    ser = pd.Series([1, 2, 3])
    method_name = 'self_accessor'
    expected_result = ser.sum() / ser.count()

    @register_series_accessor(method_name)
    def my_average(self):
        return self.sum() / self.count()
    assert method_name in pd.series._SERIES_EXTENSIONS_.keys()
    assert pd.series._SERIES_EXTENSIONS_[method_name] is my_average
    assert ser.self_accessor() == expected_result