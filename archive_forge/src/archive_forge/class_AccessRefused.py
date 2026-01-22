from struct import pack, unpack
class AccessRefused(IrrecoverableChannelError):
    """AMQP Access Refused Error."""
    code = 403