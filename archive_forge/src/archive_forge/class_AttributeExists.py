from collections import namedtuple
import re
from boto3.exceptions import DynamoDBOperationNotSupportedError
from boto3.exceptions import DynamoDBNeedsConditionError
from boto3.exceptions import DynamoDBNeedsKeyConditionError
class AttributeExists(ConditionBase):
    expression_operator = 'attribute_exists'
    expression_format = '{operator}({0})'