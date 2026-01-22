from boto.exception import BotoServerError
class OriginalTransactionIncomplete(RetriableResponseError):
    """The original transaction is still in progress.
    """