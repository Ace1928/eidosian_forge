from ...error import GraphQLError
from ...language import ast
from ...type.directives import DirectiveLocation
from .base import ValidationRule
@staticmethod
def unknown_directive_message(directive_name):
    return 'Unknown directive "{}".'.format(directive_name)