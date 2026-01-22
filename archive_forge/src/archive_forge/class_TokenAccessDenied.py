from boto.exception import BotoServerError
class TokenAccessDenied(ResponseError):
    """Permission to cancel the token is denied.
    """