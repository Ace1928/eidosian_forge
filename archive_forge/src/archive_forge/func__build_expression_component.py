from collections import namedtuple
import re
from boto3.exceptions import DynamoDBOperationNotSupportedError
from boto3.exceptions import DynamoDBNeedsConditionError
from boto3.exceptions import DynamoDBNeedsKeyConditionError
def _build_expression_component(self, value, attribute_name_placeholders, attribute_value_placeholders, has_grouped_values, is_key_condition):
    if isinstance(value, ConditionBase):
        return self._build_expression(value, attribute_name_placeholders, attribute_value_placeholders, is_key_condition)
    elif isinstance(value, AttributeBase):
        if is_key_condition and (not isinstance(value, Key)):
            raise DynamoDBNeedsKeyConditionError('Attribute object %s is of type %s. KeyConditionExpression only supports Attribute objects of type Key' % (value.name, type(value)))
        return self._build_name_placeholder(value, attribute_name_placeholders)
    else:
        return self._build_value_placeholder(value, attribute_value_placeholders, has_grouped_values)