from typing import cast, Any, Optional
from ...error import GraphQLError
from ...language import FragmentSpreadNode, InlineFragmentNode
from ...type import GraphQLCompositeType, is_composite_type
from ...utilities import do_types_overlap, type_from_ast
from . import ValidationRule
Possible fragment spread

    A fragment spread is only valid if the type condition could ever possibly be true:
    if there is a non-empty intersection of the possible parent types, and possible
    types which pass the type condition.
    