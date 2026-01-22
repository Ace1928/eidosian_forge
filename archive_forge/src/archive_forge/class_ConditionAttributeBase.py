from collections import namedtuple
import re
from boto3.exceptions import DynamoDBOperationNotSupportedError
from boto3.exceptions import DynamoDBNeedsConditionError
from boto3.exceptions import DynamoDBNeedsKeyConditionError
class ConditionAttributeBase(ConditionBase, AttributeBase):
    """This base class is for conditions that can have attribute methods.

    One example is the Size condition. To complete a condition, you need
    to apply another AttributeBase method like eq().
    """

    def __init__(self, *values):
        ConditionBase.__init__(self, *values)
        AttributeBase.__init__(self, values[0].name)

    def __eq__(self, other):
        return ConditionBase.__eq__(self, other) and AttributeBase.__eq__(self, other)

    def __ne__(self, other):
        return not self.__eq__(other)