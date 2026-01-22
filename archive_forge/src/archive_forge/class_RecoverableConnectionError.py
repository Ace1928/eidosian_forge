from struct import pack, unpack
class RecoverableConnectionError(ConnectionError):
    """Exception class for recoverable connection errors."""