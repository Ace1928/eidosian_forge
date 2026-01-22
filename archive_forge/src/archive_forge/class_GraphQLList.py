from collections.abc import Mapping, Hashable
import collections
import copy
from ..language import ast
from ..pyutils.cached_property import cached_property
from ..pyutils.ordereddict import OrderedDict
from ..utils.assert_valid_name import assert_valid_name
class GraphQLList(GraphQLType):
    """List Modifier

    A list is a kind of type marker, a wrapping type which points to another
    type. Lists are often created within the context of defining the fields
    of an object type.

    Example:

        class PersonType(GraphQLObjectType):
            name = 'Person'

            def get_fields(self):
                return {
                    'parents': GraphQLField(GraphQLList(PersonType())),
                    'children': GraphQLField(GraphQLList(PersonType())),
                }
    """
    __slots__ = ('of_type',)

    def __init__(self, type):
        assert is_type(type), 'Can only create List of a GraphQLType but got: {}.'.format(type)
        self.of_type = type

    def __str__(self):
        return '[' + str(self.of_type) + ']'

    def is_same_type(self, other):
        return isinstance(other, GraphQLList) and self.of_type.is_same_type(other.of_type)