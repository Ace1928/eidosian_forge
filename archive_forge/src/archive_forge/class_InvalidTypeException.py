from boto.exception import BotoServerError
class InvalidTypeException(BotoServerError):
    """
    Raised when an invalid record type is passed to CloudSearch.
    """
    pass