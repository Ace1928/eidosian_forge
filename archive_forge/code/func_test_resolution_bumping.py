import numpy as np
import pytest
from pandas._libs.tslibs import (
@pytest.mark.parametrize('args,expected', [((1.5, 'min'), (90, 's')), ((62.4, 'min'), (3744, 's')), ((1.04, 'h'), (3744, 's')), ((1, 'D'), (1, 'D')), ((0.342931, 'h'), (1234551600, 'us')), ((1.2345, 'D'), (106660800, 'ms'))])
def test_resolution_bumping(args, expected):
    off = to_offset(str(args[0]) + args[1])
    assert off.n == expected[0]
    assert off._prefix == expected[1]