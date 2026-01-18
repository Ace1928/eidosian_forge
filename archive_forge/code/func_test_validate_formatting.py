import re
import sys
import traceback
from typing import NoReturn
import pytest
from .._util import (
def test_validate_formatting() -> None:
    my_re = re.compile(b'foo')
    with pytest.raises(LocalProtocolError) as excinfo:
        validate(my_re, b'', 'oops')
    assert 'oops' in str(excinfo.value)
    with pytest.raises(LocalProtocolError) as excinfo:
        validate(my_re, b'', 'oops {}')
    assert 'oops {}' in str(excinfo.value)
    with pytest.raises(LocalProtocolError) as excinfo:
        validate(my_re, b'', 'oops {} xx', 10)
    assert 'oops 10 xx' in str(excinfo.value)