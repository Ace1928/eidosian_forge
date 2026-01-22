from boto.exception import BotoServerError, BotoClientError
from boto.exception import DynamoDBResponseError
class DynamoDBKeyNotFoundError(BotoClientError):
    """
    Raised when attempting to retrieve or interact with an item whose key
    can't be found.
    """
    pass