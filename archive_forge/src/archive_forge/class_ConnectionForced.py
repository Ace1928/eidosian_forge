from struct import pack, unpack
class ConnectionForced(RecoverableConnectionError):
    """AMQP Connection Forced Error."""
    code = 320