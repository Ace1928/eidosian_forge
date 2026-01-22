from functools import partial
from inspect import isclass
from ..types import Field, Interface, ObjectType
from ..types.interface import InterfaceOptions
from ..types.utils import get_type
from .id_type import BaseGlobalIDType, DefaultGlobalIDType
class AbstractNode(Interface):

    class Meta:
        abstract = True

    @classmethod
    def __init_subclass_with_meta__(cls, global_id_type=DefaultGlobalIDType, **options):
        assert issubclass(global_id_type, BaseGlobalIDType), 'Custom ID type need to be implemented as a subclass of BaseGlobalIDType.'
        _meta = InterfaceOptions(cls)
        _meta.global_id_type = global_id_type
        _meta.fields = {'id': GlobalID(cls, global_id_type=global_id_type, description='The ID of the object')}
        super(AbstractNode, cls).__init_subclass_with_meta__(_meta=_meta, **options)

    @classmethod
    def resolve_global_id(cls, info, global_id):
        return cls._meta.global_id_type.resolve_global_id(info, global_id)