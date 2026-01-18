import inspect
from pandas.util._decorators import deprecate_nonkeyword_arguments
import pandas._testing as tm
def test_three_arguments_default_allowed_args():
    with tm.assert_produces_warning(FutureWarning):
        assert g(6, 3, 3) == 12