import pytest
from winrm import Session
def test_run_cmd(protocol_fake):
    s = Session('windows-host', auth=('john.smith', 'secret'))
    s.protocol = protocol_fake
    r = s.run_cmd('ipconfig', ['/all'])
    assert r.status_code == 0
    assert b'Windows IP Configuration' in r.std_out
    assert len(r.std_err) == 0