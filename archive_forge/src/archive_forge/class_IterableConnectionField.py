import re
from collections.abc import Iterable
from functools import partial
from typing import Type
from graphql_relay import connection_from_array
from ..types import Boolean, Enum, Int, Interface, List, NonNull, Scalar, String, Union
from ..types.field import Field
from ..types.objecttype import ObjectType, ObjectTypeOptions
from ..utils.thenables import maybe_thenable
from .node import is_node, AbstractNode
class IterableConnectionField(Field):

    def __init__(self, type_, *args, **kwargs):
        kwargs.setdefault('before', String())
        kwargs.setdefault('after', String())
        kwargs.setdefault('first', Int())
        kwargs.setdefault('last', Int())
        super(IterableConnectionField, self).__init__(type_, *args, **kwargs)

    @property
    def type(self):
        type_ = super(IterableConnectionField, self).type
        connection_type = type_
        if isinstance(type_, NonNull):
            connection_type = type_.of_type
        if is_node(connection_type):
            raise Exception('ConnectionFields now need a explicit ConnectionType for Nodes.\nRead more: https://github.com/graphql-python/graphene/blob/v2.0.0/UPGRADE-v2.0.md#node-connections')
        assert issubclass(connection_type, Connection), f'{self.__class__.__name__} type has to be a subclass of Connection. Received "{connection_type}".'
        return type_

    @classmethod
    def resolve_connection(cls, connection_type, args, resolved):
        if isinstance(resolved, connection_type):
            return resolved
        assert isinstance(resolved, Iterable), f'Resolved value from the connection field has to be an iterable or instance of {connection_type}. Received "{resolved}"'
        connection = connection_from_array(resolved, args, connection_type=partial(connection_adapter, connection_type), edge_type=connection_type.Edge, page_info_type=page_info_adapter)
        connection.iterable = resolved
        return connection

    @classmethod
    def connection_resolver(cls, resolver, connection_type, root, info, **args):
        resolved = resolver(root, info, **args)
        if isinstance(connection_type, NonNull):
            connection_type = connection_type.of_type
        on_resolve = partial(cls.resolve_connection, connection_type, args)
        return maybe_thenable(resolved, on_resolve)

    def wrap_resolve(self, parent_resolver):
        resolver = super(IterableConnectionField, self).wrap_resolve(parent_resolver)
        return partial(self.connection_resolver, resolver, self.type)