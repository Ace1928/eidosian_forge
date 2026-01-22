from boto.exception import BotoServerError
class InvalidTransactionState(ResponseError):
    """The transaction is not complete, or it has temporarily failed.
    """