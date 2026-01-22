import itertools
from collections import OrderedDict
from ...error import GraphQLError
from ...language import ast
from ...language.printer import print_ast
from ...pyutils.pair_set import PairSet
from ...type.definition import (GraphQLInterfaceType, GraphQLList,
from ...utils.type_comparators import is_equal_type
from ...utils.type_from_ast import type_from_ast
from .base import ValidationRule
def _collect_conflicts_within(context, conflicts, cached_fields_and_fragment_names, compared_fragments, field_map):
    """Collect all Conflicts "within" one collection of fields."""
    for response_name, fields in list(field_map.items()):
        for i, field in enumerate(fields):
            for other_field in fields[i + 1:]:
                conflict = _find_conflict(context, cached_fields_and_fragment_names, compared_fragments, False, response_name, field, other_field)
                if conflict:
                    conflicts.append(conflict)