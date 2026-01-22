from boto.exception import BotoServerError
class InvalidTokenId(ResponseError):
    """You did not install the token that you are trying to cancel.
    """