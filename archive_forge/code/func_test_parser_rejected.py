import pytest
from jeepney import auth
def test_parser_rejected():
    p = auth.SASLParser()
    with pytest.raises(auth.AuthenticationError):
        p.feed(b'REJECTED EXTERNAL\r\n')
    assert not p.authenticated