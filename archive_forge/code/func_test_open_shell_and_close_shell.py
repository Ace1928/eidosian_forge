import pytest
from winrm.protocol import Protocol
def test_open_shell_and_close_shell(protocol_fake):
    shell_id = protocol_fake.open_shell()
    assert shell_id == '11111111-1111-1111-1111-111111111113'
    protocol_fake.close_shell(shell_id, close_session=True)