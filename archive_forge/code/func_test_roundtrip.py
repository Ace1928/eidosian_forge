import pytest
import subprocess
import json
import sys
from numpy.distutils import _shell_utils
from numpy.testing import IS_WASM
@pytest.mark.skipif(IS_WASM, reason='Cannot start subprocess')
@pytest.mark.parametrize('argv', argv_cases)
def test_roundtrip(Parser, argv):
    """
    Test that split is the inverse operation of join
    """
    try:
        joined = Parser.join(argv)
        assert argv == Parser.split(joined)
    except NotImplementedError:
        pytest.skip('Not implemented')