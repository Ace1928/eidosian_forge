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
def _find_conflicts_within_selection_set(context, cached_fields_and_fragment_names, compared_fragments, parent_type, selection_set):
    """Find all conflicts found "within" a selection set, including those found via spreading in fragments.

       Called when visiting each SelectionSet in the GraphQL Document.
    """
    conflicts = []
    field_map, fragment_names = _get_fields_and_fragments_names(context, cached_fields_and_fragment_names, parent_type, selection_set)
    _collect_conflicts_within(context, conflicts, cached_fields_and_fragment_names, compared_fragments, field_map)
    for i, fragment_name in enumerate(fragment_names):
        _collect_conflicts_between_fields_and_fragment(context, conflicts, cached_fields_and_fragment_names, compared_fragments, False, field_map, fragment_name)
        for other_fragment_name in fragment_names[i + 1:]:
            _collect_conflicts_between_fragments(context, conflicts, cached_fields_and_fragment_names, compared_fragments, False, fragment_name, other_fragment_name)
    return conflicts