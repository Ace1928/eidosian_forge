from boto.exception import BotoServerError
class IncompatibleTokens(ResponseError):
    """The transaction could not be completed because the tokens have
       incompatible payment instructions.
    """