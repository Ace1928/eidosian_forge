from boto.exception import BotoServerError
class InvalidAccountState_Recipient(RetriableResponseError):
    """Recipient account cannot participate in the transaction.
    """