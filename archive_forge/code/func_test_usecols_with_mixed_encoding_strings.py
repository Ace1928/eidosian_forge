from io import StringIO
import pytest
from pandas import DataFrame
import pandas._testing as tm
@pytest.mark.parametrize('usecols', [['AAA', b'BBB'], [b'AAA', 'BBB']])
def test_usecols_with_mixed_encoding_strings(all_parsers, usecols):
    data = 'AAA,BBB,CCC,DDD\n0.056674973,8,True,a\n2.613230982,2,False,b\n3.568935038,7,False,a'
    parser = all_parsers
    _msg_validate_usecols_arg = "'usecols' must either be list-like of all strings, all unicode, all integers or a callable."
    with pytest.raises(ValueError, match=_msg_validate_usecols_arg):
        parser.read_csv(StringIO(data), usecols=usecols)