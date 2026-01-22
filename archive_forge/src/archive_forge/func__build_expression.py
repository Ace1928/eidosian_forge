from collections import namedtuple
import re
from boto3.exceptions import DynamoDBOperationNotSupportedError
from boto3.exceptions import DynamoDBNeedsConditionError
from boto3.exceptions import DynamoDBNeedsKeyConditionError
def _build_expression(self, condition, attribute_name_placeholders, attribute_value_placeholders, is_key_condition):
    expression_dict = condition.get_expression()
    replaced_values = []
    for value in expression_dict['values']:
        replaced_value = self._build_expression_component(value, attribute_name_placeholders, attribute_value_placeholders, condition.has_grouped_values, is_key_condition)
        replaced_values.append(replaced_value)
    return expression_dict['format'].format(*replaced_values, operator=expression_dict['operator'])