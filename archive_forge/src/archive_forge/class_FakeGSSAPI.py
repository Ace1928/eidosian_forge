import socket
import warnings
from io import BytesIO
from amqp.serialization import _write_table
class FakeGSSAPI(SASL):
    """A no-op SASL mechanism for when gssapi isn't available."""
    mechanism = None

    def __init__(self, client_name=None, service=b'amqp', rdns=False, fail_soft=False):
        if not fail_soft:
            raise NotImplementedError('You need to install the `gssapi` module for GSSAPI SASL support')

    def start(self):
        return NotImplemented