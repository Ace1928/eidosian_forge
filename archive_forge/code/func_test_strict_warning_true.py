import os
from skimage._shared._warnings import expected_warnings
import pytest
@pytest.mark.parametrize('strictness', ['1', 'true', 'True', 'TRUE'])
def test_strict_warning_true(setup, strictness):
    os.environ['SKIMAGE_TEST_STRICT_WARNINGS'] = strictness
    with pytest.raises(ValueError):
        with expected_warnings(['some warnings']):
            pass