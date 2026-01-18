import sys
import types
import pytest
from ..sexts import package_check
def test_package_check():
    with pytest.raises(RuntimeError):
        package_check(FAKE_NAME)
    package_check(FAKE_NAME, optional=True)
    package_check(FAKE_NAME, optional='some-package')
    try:
        sys.modules[FAKE_NAME] = FAKE_MODULE
        package_check(FAKE_NAME)
        FAKE_MODULE.__version__ = '0.2'
        package_check(FAKE_NAME, version='0.2')
        with pytest.raises(RuntimeError):
            package_check(FAKE_NAME, '0.3')
        package_check(FAKE_NAME, version='0.3', optional=True)
        package_check(FAKE_NAME, version='0.2', version_getter=lambda x: '0.2')
    finally:
        del sys.modules[FAKE_NAME]