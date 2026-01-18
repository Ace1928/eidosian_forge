from unittest import TestCase
from zmq.utils import z85
def test_server_secret(self):
    server_secret = b'\x8e\x0b\xddiv(\xb9\x1d\x8f$U\x87\xee\x95\xc5\xb0MH\x96?y%\x98w\xb4\x9c\xd9\x06:\xea\xd3\xb7'
    encoded = z85.encode(server_secret)
    assert encoded == b'JTKVSB%%)wK0E.X)V>+}o?pNmC{O&4W4b!Ni{Lh6'
    decoded = z85.decode(encoded)
    assert decoded == server_secret