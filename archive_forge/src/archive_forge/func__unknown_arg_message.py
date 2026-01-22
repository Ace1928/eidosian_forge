from ...error import GraphQLError
from ...language import ast
from ...utils.quoted_or_list import quoted_or_list
from ...utils.suggestion_list import suggestion_list
from .base import ValidationRule
def _unknown_arg_message(arg_name, field_name, type, suggested_args):
    message = 'Unknown argument "{}" on field "{}" of type "{}".'.format(arg_name, field_name, type)
    if suggested_args:
        message += ' Did you mean {}?'.format(quoted_or_list(suggested_args))
    return message