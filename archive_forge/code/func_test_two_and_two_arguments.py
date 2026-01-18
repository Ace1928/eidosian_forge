import inspect
from pandas.util._decorators import deprecate_nonkeyword_arguments
import pandas._testing as tm
def test_two_and_two_arguments():
    with tm.assert_produces_warning(None):
        assert f(1, 3, c=3, d=5) == 12