from ...error import GraphQLError
from .base import ValidationRule
@staticmethod
def unused_variable_message(variable_name, op_name):
    if op_name:
        return 'Variable "${}" is never used in operation "{}".'.format(variable_name, op_name)
    return 'Variable "${}" is never used.'.format(variable_name)