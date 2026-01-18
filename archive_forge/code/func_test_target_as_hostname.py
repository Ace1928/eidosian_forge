import pytest
from winrm import Session
def test_target_as_hostname():
    s = Session('windows-host', auth=('john.smith', 'secret'))
    assert s.url == 'http://windows-host:5985/wsman'