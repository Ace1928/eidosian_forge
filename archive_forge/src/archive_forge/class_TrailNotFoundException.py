from boto.exception import BotoServerError
class TrailNotFoundException(BotoServerError):
    """
    Raised when the given trail name is not found.
    """
    pass