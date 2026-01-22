from typing import Any, Collection, Dict, Optional, Tuple, cast
from ..language import ast, DirectiveLocation
from ..pyutils import inspect, is_description
from .assert_name import assert_name
from .definition import GraphQLArgument, GraphQLInputType, GraphQLNonNull, is_input_type
from .scalars import GraphQLBoolean, GraphQLString
class GraphQLDirectiveKwargs(TypedDict, total=False):
    name: str
    locations: Tuple[DirectiveLocation, ...]
    args: Dict[str, GraphQLArgument]
    is_repeatable: bool
    description: Optional[str]
    extensions: Dict[str, Any]
    ast_node: Optional[ast.DirectiveDefinitionNode]