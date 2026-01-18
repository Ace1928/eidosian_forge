from graphene.types.enum import Enum, EnumOptions
from graphene.types.inputobjecttype import InputObjectType
from graphene.types.objecttype import ObjectType, ObjectTypeOptions
def test_special_enum_could_be_subclassed_default():

    class MyEnum(SpecialEnum):
        pass
    assert MyEnum._meta.other_attr == 'default'