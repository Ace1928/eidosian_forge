from struct import pack, unpack
class UnexpectedFrame(IrrecoverableConnectionError):
    """AMQP Unexpected Frame."""
    code = 505