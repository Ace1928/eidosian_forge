import pytest
from pandas.util._validators import (
@pytest.mark.parametrize('name', ['inplace', 'copy'])
@pytest.mark.parametrize('value', [1, 'True', [1, 2, 3], 5.0])
def test_validate_bool_kwarg_fail(name, value):
    msg = f'For argument "{name}" expected type bool, received type {type(value).__name__}'
    with pytest.raises(ValueError, match=msg):
        validate_bool_kwarg(value, name)