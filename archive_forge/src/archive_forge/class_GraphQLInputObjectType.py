from collections.abc import Mapping, Hashable
import collections
import copy
from ..language import ast
from ..pyutils.cached_property import cached_property
from ..pyutils.ordereddict import OrderedDict
from ..utils.assert_valid_name import assert_valid_name
class GraphQLInputObjectType(GraphQLType):
    """Input Object Type Definition

    An input object defines a structured collection of fields which may be
    supplied to a field argument.

    Using `NonNull` will ensure that a value must be provided by the query

    Example:

        NonNullFloat = GraphQLNonNull(GraphQLFloat())

        class GeoPoint(GraphQLInputObjectType):
            name = 'GeoPoint'
            fields = {
                'lat': GraphQLInputObjectField(NonNullFloat),
                'lon': GraphQLInputObjectField(NonNullFloat),
                'alt': GraphQLInputObjectField(GraphQLFloat(),
                    default_value=0)
            }
    """

    def __init__(self, name, fields, description=None):
        assert name, 'Type must be named.'
        self.name = name
        self.description = description
        self._fields = fields

    @cached_property
    def fields(self):
        return self._define_field_map()

    def _define_field_map(self):
        fields = self._fields
        if callable(fields):
            fields = fields()
        assert isinstance(fields, Mapping) and len(fields) > 0, '{} fields must be a mapping (dict / OrderedDict) with field names as keys or a function which returns such a mapping.'.format(self)
        if not isinstance(fields, (collections.OrderedDict, OrderedDict)):
            fields = OrderedDict(sorted(list(fields.items())))
        for field_name, field in fields.items():
            assert_valid_name(field_name)
        return fields