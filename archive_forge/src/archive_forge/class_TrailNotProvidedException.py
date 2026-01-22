from boto.exception import BotoServerError
class TrailNotProvidedException(BotoServerError):
    """
    Raised when no trail name was provided.
    """
    pass