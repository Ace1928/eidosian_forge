from graphene.types.enum import Enum, EnumOptions
from graphene.types.inputobjecttype import InputObjectType
from graphene.types.objecttype import ObjectType, ObjectTypeOptions
def test_special_inputobjecttype_inherit_meta_options():

    class MyInputObjectType(SpecialInputObjectType):
        pass
    assert MyInputObjectType._meta.name == 'MyInputObjectType'