from collections.abc import Mapping, Hashable
import collections
import copy
from ..language import ast
from ..pyutils.cached_property import cached_property
from ..pyutils.ordereddict import OrderedDict
from ..utils.assert_valid_name import assert_valid_name
class GraphQLField(object):
    __slots__ = ('type', 'args', 'resolver', 'deprecation_reason', 'description')

    def __init__(self, type, args=None, resolver=None, deprecation_reason=None, description=None):
        self.type = type
        self.args = args or OrderedDict()
        self.resolver = resolver
        self.deprecation_reason = deprecation_reason
        self.description = description

    def __eq__(self, other):
        return self is other or (isinstance(other, GraphQLField) and self.type == other.type and (self.args == other.args) and (self.resolver == other.resolver) and (self.deprecation_reason == other.deprecation_reason) and (self.description == other.description))

    def __hash__(self):
        return id(self)