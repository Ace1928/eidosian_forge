import pytest
from IPython.core.prefilter import AutocallChecker
def test_prefilter():
    """Test user input conversions"""
    pairs = [('2+2', '2+2')]
    for raw, correct in pairs:
        assert ip.prefilter(raw) == correct