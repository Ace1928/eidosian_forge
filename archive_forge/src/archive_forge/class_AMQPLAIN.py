import socket
import warnings
from io import BytesIO
from amqp.serialization import _write_table
class AMQPLAIN(SASL):
    """AMQPLAIN SASL authentication mechanism.

    This is a non-standard mechanism used by AMQP servers.
    """
    mechanism = b'AMQPLAIN'

    def __init__(self, username, password):
        self.username, self.password = (username, password)
    __slots__ = ('username', 'password')

    def start(self, connection):
        if self.username is None or self.password is None:
            return NotImplemented
        login_response = BytesIO()
        _write_table({b'LOGIN': self.username, b'PASSWORD': self.password}, login_response.write, [])
        return login_response.getvalue()[4:]