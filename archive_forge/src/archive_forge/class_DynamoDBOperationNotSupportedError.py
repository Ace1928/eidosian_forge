import botocore.exceptions
class DynamoDBOperationNotSupportedError(Boto3Error):
    """Raised for operations that are not supported for an operand."""

    def __init__(self, operation, value):
        msg = '%s operation cannot be applied to value %s of type %s directly. Must use AttributeBase object methods (i.e. Attr().eq()). to generate ConditionBase instances first.' % (operation, value, type(value))
        Exception.__init__(self, msg)