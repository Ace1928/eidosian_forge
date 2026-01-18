import inspect
from pandas.util._decorators import deprecate_nonkeyword_arguments
import pandas._testing as tm
def test_one_positional_argument_with_warning_message_analysis():
    msg = 'Starting with pandas version 1.1 all arguments of h will be keyword-only.'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        assert h(19) == 19