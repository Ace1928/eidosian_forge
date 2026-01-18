from typing import cast, overload, Optional
from ..language import ListTypeNode, NamedTypeNode, NonNullTypeNode, TypeNode
from ..pyutils import inspect
from ..type import (
Get the GraphQL type definition from an AST node.

    Given a Schema and an AST node describing a type, return a GraphQLType definition
    which applies to that type. For example, if provided the parsed AST node for
    ``[User]``, a GraphQLList instance will be returned, containing the type called
    "User" found in the schema. If a type called "User" is not found in the schema,
    then None will be returned.
    