from boto.exception import BotoServerError
class SESAddressNotVerifiedError(SESError):
    """
    Raised when a "Reply-To" address has not been validated in SES yet.
    """
    pass