from boto.exception import BotoServerError
class InternalException(BotoServerError):
    """
    A generic server-side error.
    """
    pass