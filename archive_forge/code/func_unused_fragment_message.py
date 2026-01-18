from ...error import GraphQLError
from .base import ValidationRule
@staticmethod
def unused_fragment_message(fragment_name):
    return 'Fragment "{}" is never used.'.format(fragment_name)