from collections import defaultdict
from ..error import GraphQLError
from ..language import ast
from ..pyutils.ordereddict import OrderedDict
from ..type.definition import (GraphQLArgument, GraphQLEnumType,
from ..type.introspection import (__Directive, __DirectiveLocation,
from ..type.scalars import (GraphQLBoolean, GraphQLFloat, GraphQLID,
from ..type.schema import GraphQLSchema
from .value_from_ast import value_from_ast
Produces a new schema given an existing schema and a document which may
    contain GraphQL type extensions and definitions. The original schema will
    remain unaltered.

    Because a schema represents a graph of references, a schema cannot be
    extended without effectively making an entire copy. We do not know until it's
    too late if subgraphs remain unchanged.

    This algorithm copies the provided schema, applying extensions while
    producing the copy. The original schema remains unaltered.