import inspect
from pandas.util._decorators import deprecate_nonkeyword_arguments
import pandas._testing as tm
def test_three_arguments_with_name_in_warning():
    msg = "Starting with pandas version 1.1 all arguments of f_add_inputs except for the arguments 'a' and 'b' will be keyword-only."
    with tm.assert_produces_warning(FutureWarning, match=msg):
        assert f(6, 3, 3) == 12