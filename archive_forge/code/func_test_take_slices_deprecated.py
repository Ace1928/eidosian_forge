import pytest
import pandas._testing as tm
def test_take_slices_deprecated(self, float_frame):
    df = float_frame
    slc = slice(0, 4, 1)
    with tm.assert_produces_warning(FutureWarning):
        df.take(slc, axis=0)
    with tm.assert_produces_warning(FutureWarning):
        df.take(slc, axis=1)