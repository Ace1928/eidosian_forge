import h2.errors
class NoAvailableStreamIDError(ProtocolError):
    """
    There are no available stream IDs left to the connection. All stream IDs
    have been exhausted.

    .. versionadded:: 2.0.0
    """
    pass