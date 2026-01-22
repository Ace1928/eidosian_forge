import botocore.exceptions
class DynamoDBNeedsConditionError(Boto3Error):
    """Raised when input is not a condition"""

    def __init__(self, value):
        msg = 'Expecting a ConditionBase object. Got %s of type %s. Use AttributeBase object methods (i.e. Attr().eq()). to generate ConditionBase instances.' % (value, type(value))
        Exception.__init__(self, msg)