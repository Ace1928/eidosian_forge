from datetime import datetime
from pandas import Series
def test_set_name_attribute(self):
    ser = Series([1, 2, 3])
    ser2 = Series([1, 2, 3], name='bar')
    for name in [7, 7.0, 'name', datetime(2001, 1, 1), (1,), 'א']:
        ser.name = name
        assert ser.name == name
        ser2.name = name
        assert ser2.name == name