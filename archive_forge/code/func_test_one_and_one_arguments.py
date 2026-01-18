import inspect
from pandas.util._decorators import deprecate_nonkeyword_arguments
import pandas._testing as tm
def test_one_and_one_arguments():
    with tm.assert_produces_warning(None):
        assert f(19, d=6) == 25