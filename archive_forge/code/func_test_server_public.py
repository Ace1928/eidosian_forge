from unittest import TestCase
from zmq.utils import z85
def test_server_public(self):
    server_public = b"T\xfc\xba$\xe92I\x96\x93\x16\xfba|\x87+\xb0\xc1\xd1\xff\x14\x80\x04'\xc5\x94\xcb\xfa\xcf\x1b\xc2\xd6R"
    encoded = z85.encode(server_public)
    assert encoded == b'rq:rM>}U?@Lns47E1%kR.o@n%FcmmsL/@{H8]yf7'
    decoded = z85.decode(encoded)
    assert decoded == server_public