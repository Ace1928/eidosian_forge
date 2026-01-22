from copy import copy, deepcopy
from typing import (
from ..error import GraphQLError
from ..language import ast, OperationType
from ..pyutils import inspect, is_collection, is_description
from .definition import (
from .directives import GraphQLDirective, specified_directives, is_directive
from .introspection import introspection_types
class GraphQLSchemaKwargs(TypedDict, total=False):
    query: Optional[GraphQLObjectType]
    mutation: Optional[GraphQLObjectType]
    subscription: Optional[GraphQLObjectType]
    types: Optional[Tuple[GraphQLNamedType, ...]]
    directives: Tuple[GraphQLDirective, ...]
    description: Optional[str]
    extensions: Dict[str, Any]
    ast_node: Optional[ast.SchemaDefinitionNode]
    extension_ast_nodes: Tuple[ast.SchemaExtensionNode, ...]
    assume_valid: bool