from boto.exception import BotoServerError
class InvalidClientTokenId(ResponseError):
    """The AWS Access Key Id you provided does not exist in our records.
    """