from boto.exception import BotoServerError, BotoClientError
from boto.exception import DynamoDBResponseError
class DynamoDBValidationError(DynamoDBResponseError):
    """
    Raised when a ValidationException response is received. This happens
    when one or more required parameter values are missing, or if the item
    has exceeded the 64Kb size limit.
    """
    pass