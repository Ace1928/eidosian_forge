import pytest
from ase.io.cif import CIFBlock, parse_loop, CIFLoop
def test_parse_cifloop_warn_duplicate_header():
    with pytest.warns(UserWarning):
        parse_loop(['_hello', '_hello'])