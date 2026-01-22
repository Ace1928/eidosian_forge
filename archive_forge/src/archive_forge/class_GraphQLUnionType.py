from collections.abc import Mapping, Hashable
import collections
import copy
from ..language import ast
from ..pyutils.cached_property import cached_property
from ..pyutils.ordereddict import OrderedDict
from ..utils.assert_valid_name import assert_valid_name
class GraphQLUnionType(GraphQLType):
    """Union Type Definition

    When a field can return one of a heterogeneous set of types, a Union type is used to describe what types are possible
    as well as providing a function to determine which type is actually used when the field is resolved.

    Example:

        class PetType(GraphQLUnionType):
            name = 'Pet'
            types = [DogType, CatType]

            def resolve_type(self, value):
                if isinstance(value, Dog):
                    return DogType()
                if isinstance(value, Cat):
                    return CatType()
    """

    def __init__(self, name, types=None, resolve_type=None, description=None):
        assert name, 'Type must be named.'
        assert_valid_name(name)
        self.name = name
        self.description = description
        if resolve_type is not None:
            assert callable(resolve_type), '{} must provide "resolve_type" as a function.'.format(self)
        self.resolve_type = resolve_type
        self._types = types

    @cached_property
    def types(self):
        return define_types(self, self._types)