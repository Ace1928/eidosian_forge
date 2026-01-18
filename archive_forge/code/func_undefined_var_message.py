from ...error import GraphQLError
from .base import ValidationRule
@staticmethod
def undefined_var_message(var_name, op_name=None):
    if op_name:
        return 'Variable "${}" is not defined by operation "{}".'.format(var_name, op_name)
    return 'Variable "${}" is not defined.'.format(var_name)