import pytest
from winrm import Session
def test_target_with_dots():
    s = Session('windows-host.example.com', auth=('john.smith', 'secret'))
    assert s.url == 'http://windows-host.example.com:5985/wsman'