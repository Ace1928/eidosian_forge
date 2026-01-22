from functools import partial
from inspect import isclass
from ..types import Field, Interface, ObjectType
from ..types.interface import InterfaceOptions
from ..types.utils import get_type
from .id_type import BaseGlobalIDType, DefaultGlobalIDType
class NodeField(Field):

    def __init__(self, node, type_=False, **kwargs):
        assert issubclass(node, Node), 'NodeField can only operate in Nodes'
        self.node_type = node
        self.field_type = type_
        global_id_type = node._meta.global_id_type
        super(NodeField, self).__init__(type_ or node, id=global_id_type.graphene_type(required=True, description='The ID of the object'), **kwargs)

    def wrap_resolve(self, parent_resolver):
        return partial(self.node_type.node_resolver, get_type(self.field_type))