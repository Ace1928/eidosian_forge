from zaqarclient import errors
class MalformedRequest(TransportError):
    """Indicates that a request is malformed

    This error maps to HTTP's 400
    """
    code = 400