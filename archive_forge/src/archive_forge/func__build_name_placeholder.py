from collections import namedtuple
import re
from boto3.exceptions import DynamoDBOperationNotSupportedError
from boto3.exceptions import DynamoDBNeedsConditionError
from boto3.exceptions import DynamoDBNeedsKeyConditionError
def _build_name_placeholder(self, value, attribute_name_placeholders):
    attribute_name = value.name
    attribute_name_parts = ATTR_NAME_REGEX.findall(attribute_name)
    placeholder_format = ATTR_NAME_REGEX.sub('%s', attribute_name)
    str_format_args = []
    for part in attribute_name_parts:
        name_placeholder = self._get_name_placeholder()
        self._name_count += 1
        str_format_args.append(name_placeholder)
        attribute_name_placeholders[name_placeholder] = part
    return placeholder_format % tuple(str_format_args)