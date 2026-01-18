from graphene.types.enum import Enum, EnumOptions
from graphene.types.inputobjecttype import InputObjectType
from graphene.types.objecttype import ObjectType, ObjectTypeOptions
def test_special_objecttype_inherit_meta_options():

    class MyType(SpecialObjectType):
        pass
    assert MyType._meta.name == 'MyType'
    assert MyType._meta.default_resolver is None
    assert MyType._meta.interfaces == ()