import sys
import types
import pytest
from ..sexts import package_check
def test_package_check_setuptools():
    with pytest.raises(RuntimeError):
        package_check(FAKE_NAME, setuptools_args=None)

    def pkg_chk_sta(*args, **kwargs):
        st_args = {}
        package_check(*args, setuptools_args=st_args, **kwargs)
        return st_args
    assert pkg_chk_sta(FAKE_NAME) == {'install_requires': ['nisext_improbable']}
    old_sta = {'install_requires': ['something']}
    package_check(FAKE_NAME, setuptools_args=old_sta)
    assert old_sta == {'install_requires': ['something', 'nisext_improbable']}
    old_sta = {'install_requires': 'something'}
    package_check(FAKE_NAME, setuptools_args=old_sta)
    assert old_sta == {'install_requires': ['something', 'nisext_improbable']}
    assert pkg_chk_sta(FAKE_NAME, optional='something') == {'extras_require': {'something': ['nisext_improbable']}}
    old_sta = {'extras_require': {'something': ['amodule']}}
    package_check(FAKE_NAME, optional='something', setuptools_args=old_sta)
    assert old_sta == {'extras_require': {'something': ['amodule', 'nisext_improbable']}}
    old_sta = {'extras_require': {'something': 'amodule'}}
    package_check(FAKE_NAME, optional='something', setuptools_args=old_sta)
    assert old_sta == {'extras_require': {'something': ['amodule', 'nisext_improbable']}}
    with pytest.raises(RuntimeError):
        package_check(FAKE_NAME, optional=True, setuptools_args={})
    try:
        sys.modules[FAKE_NAME] = FAKE_MODULE
        assert pkg_chk_sta(FAKE_NAME) == {}
        FAKE_MODULE.__version__ = '0.2'
        assert pkg_chk_sta(FAKE_NAME, version='0.2') == {}
        exp_spec = [FAKE_NAME + '>=0.3']
        assert pkg_chk_sta(FAKE_NAME, version='0.3') == {'install_requires': exp_spec}
        package_check(FAKE_NAME, version='0.2', version_getter=lambda x: '0.2')
        assert pkg_chk_sta(FAKE_NAME, version='0.3', optional='afeature') == {'extras_require': {'afeature': exp_spec}}
        assert pkg_chk_sta(FAKE_NAME, version='0.2', version_getter=lambda x: '0.2') == {}
        bad_getter = lambda x: x.not_an_attribute
        exp_spec = [FAKE_NAME + '>=0.2']
        assert pkg_chk_sta(FAKE_NAME, version='0.2', version_getter=bad_getter) == {'install_requires': exp_spec}
        assert pkg_chk_sta(FAKE_NAME, version='0.2', optional='afeature', version_getter=bad_getter) == {'extras_require': {'afeature': [FAKE_NAME + '>=0.2']}}
    finally:
        del sys.modules[FAKE_NAME]