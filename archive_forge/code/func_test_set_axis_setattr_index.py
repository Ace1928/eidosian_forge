from datetime import datetime
import pytz
from pandas import DataFrame
import pandas._testing as tm
def test_set_axis_setattr_index(self):
    df = DataFrame([{'ts': datetime(2014, 4, 1, tzinfo=pytz.utc), 'foo': 1}])
    expected = df.set_index('ts')
    df.index = df['ts']
    df.pop('ts')
    tm.assert_frame_equal(df, expected)