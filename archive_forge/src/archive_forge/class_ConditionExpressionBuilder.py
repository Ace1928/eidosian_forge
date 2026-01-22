from collections import namedtuple
import re
from boto3.exceptions import DynamoDBOperationNotSupportedError
from boto3.exceptions import DynamoDBNeedsConditionError
from boto3.exceptions import DynamoDBNeedsKeyConditionError
class ConditionExpressionBuilder(object):
    """This class is used to build condition expressions with placeholders"""

    def __init__(self):
        self._name_count = 0
        self._value_count = 0
        self._name_placeholder = 'n'
        self._value_placeholder = 'v'

    def _get_name_placeholder(self):
        return '#' + self._name_placeholder + str(self._name_count)

    def _get_value_placeholder(self):
        return ':' + self._value_placeholder + str(self._value_count)

    def reset(self):
        """Resets the placeholder name and values"""
        self._name_count = 0
        self._value_count = 0

    def build_expression(self, condition, is_key_condition=False):
        """Builds the condition expression and the dictionary of placeholders.

        :type condition: ConditionBase
        :param condition: A condition to be built into a condition expression
            string with any necessary placeholders.

        :type is_key_condition: Boolean
        :param is_key_condition: True if the expression is for a
            KeyConditionExpression. False otherwise.

        :rtype: (string, dict, dict)
        :returns: Will return a string representing the condition with
            placeholders inserted where necessary, a dictionary of
            placeholders for attribute names, and a dictionary of
            placeholders for attribute values. Here is a sample return value:

            ('#n0 = :v0', {'#n0': 'myattribute'}, {':v1': 'myvalue'})
        """
        if not isinstance(condition, ConditionBase):
            raise DynamoDBNeedsConditionError(condition)
        attribute_name_placeholders = {}
        attribute_value_placeholders = {}
        condition_expression = self._build_expression(condition, attribute_name_placeholders, attribute_value_placeholders, is_key_condition=is_key_condition)
        return BuiltConditionExpression(condition_expression=condition_expression, attribute_name_placeholders=attribute_name_placeholders, attribute_value_placeholders=attribute_value_placeholders)

    def _build_expression(self, condition, attribute_name_placeholders, attribute_value_placeholders, is_key_condition):
        expression_dict = condition.get_expression()
        replaced_values = []
        for value in expression_dict['values']:
            replaced_value = self._build_expression_component(value, attribute_name_placeholders, attribute_value_placeholders, condition.has_grouped_values, is_key_condition)
            replaced_values.append(replaced_value)
        return expression_dict['format'].format(*replaced_values, operator=expression_dict['operator'])

    def _build_expression_component(self, value, attribute_name_placeholders, attribute_value_placeholders, has_grouped_values, is_key_condition):
        if isinstance(value, ConditionBase):
            return self._build_expression(value, attribute_name_placeholders, attribute_value_placeholders, is_key_condition)
        elif isinstance(value, AttributeBase):
            if is_key_condition and (not isinstance(value, Key)):
                raise DynamoDBNeedsKeyConditionError('Attribute object %s is of type %s. KeyConditionExpression only supports Attribute objects of type Key' % (value.name, type(value)))
            return self._build_name_placeholder(value, attribute_name_placeholders)
        else:
            return self._build_value_placeholder(value, attribute_value_placeholders, has_grouped_values)

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

    def _build_value_placeholder(self, value, attribute_value_placeholders, has_grouped_values=False):
        if has_grouped_values:
            placeholder_list = []
            for v in value:
                value_placeholder = self._get_value_placeholder()
                self._value_count += 1
                placeholder_list.append(value_placeholder)
                attribute_value_placeholders[value_placeholder] = v
            return '(' + ', '.join(placeholder_list) + ')'
        else:
            value_placeholder = self._get_value_placeholder()
            self._value_count += 1
            attribute_value_placeholders[value_placeholder] = value
            return value_placeholder