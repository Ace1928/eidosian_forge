from collections import Counter
from ...error import GraphQLError
from ...pyutils.ordereddict import OrderedDict
from ...type.definition import (GraphQLInterfaceType, GraphQLObjectType,
from ...utils.quoted_or_list import quoted_or_list
from ...utils.suggestion_list import suggestion_list
from .base import ValidationRule
def _undefined_field_message(field_name, type, suggested_types, suggested_fields):
    message = 'Cannot query field "{}" on type "{}".'.format(field_name, type)
    if suggested_types:
        suggestions = quoted_or_list(suggested_types)
        message += ' Did you mean to use an inline fragment on {}?'.format(suggestions)
    elif suggested_fields:
        suggestions = quoted_or_list(suggested_fields)
        message += ' Did you mean {}?'.format(suggestions)
    return message