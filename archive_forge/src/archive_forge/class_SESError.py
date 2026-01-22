from boto.exception import BotoServerError
class SESError(BotoServerError):
    """
    Sub-class all SES-related errors from here. Don't raise this error
    directly from anywhere. The only thing this gets us is the ability to
    catch SESErrors separately from the more generic, top-level
    BotoServerError exception.
    """
    pass