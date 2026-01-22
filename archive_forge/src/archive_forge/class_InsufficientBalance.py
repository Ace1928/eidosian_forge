from boto.exception import BotoServerError
class InsufficientBalance(RetriableResponseError):
    """The sender, caller, or recipient's account balance has
       insufficient funds to complete the transaction.
    """