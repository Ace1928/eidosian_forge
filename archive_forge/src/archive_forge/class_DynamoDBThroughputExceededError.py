from boto.exception import BotoServerError, BotoClientError
from boto.exception import DynamoDBResponseError
class DynamoDBThroughputExceededError(DynamoDBResponseError):
    """
    Raised when the provisioned throughput has been exceeded.
    Normally, when provisioned throughput is exceeded the operation
    is retried.  If the retries are exhausted then this exception
    will be raised.
    """
    pass