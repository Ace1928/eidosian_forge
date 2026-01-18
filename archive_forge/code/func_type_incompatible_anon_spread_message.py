from ...error import GraphQLError
from ...utils.type_comparators import do_types_overlap
from ...utils.type_from_ast import type_from_ast
from .base import ValidationRule
@staticmethod
def type_incompatible_anon_spread_message(parent_type, frag_type):
    return 'Fragment cannot be spread here as objects of type {} can never be of type {}'.format(parent_type, frag_type)