import h2.errors
class StreamClosedError(NoSuchStreamError):
    """
    A more specific form of
    :class:`NoSuchStreamError <h2.exceptions.NoSuchStreamError>`. Indicates
    that the stream has since been closed, and that all state relating to that
    stream has been removed.
    """

    def __init__(self, stream_id):
        self.stream_id = stream_id
        self.error_code = h2.errors.ErrorCodes.STREAM_CLOSED
        self._events = []