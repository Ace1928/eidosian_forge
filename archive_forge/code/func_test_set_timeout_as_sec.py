import pytest
from winrm.protocol import Protocol
def test_set_timeout_as_sec():
    protocol = Protocol('endpoint', username='username', password='password', read_timeout_sec='30', operation_timeout_sec='29')
    assert protocol.read_timeout_sec == 30
    assert protocol.operation_timeout_sec == 29