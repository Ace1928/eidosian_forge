from graphene.types.enum import Enum, EnumOptions
from graphene.types.inputobjecttype import InputObjectType
from graphene.types.objecttype import ObjectType, ObjectTypeOptions
class SpecialObjectType(ObjectType):

    @classmethod
    def __init_subclass_with_meta__(cls, other_attr='default', **options):
        _meta = SpecialOptions(cls)
        _meta.other_attr = other_attr
        super(SpecialObjectType, cls).__init_subclass_with_meta__(_meta=_meta, **options)