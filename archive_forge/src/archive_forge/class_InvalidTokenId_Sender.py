from boto.exception import BotoServerError
class InvalidTokenId_Sender(ResponseError):
    """The sender token specified is either invalid or canceled or the
       token is not active.
    """