from graphene.types.enum import Enum, EnumOptions
from graphene.types.inputobjecttype import InputObjectType
from graphene.types.objecttype import ObjectType, ObjectTypeOptions
def test_special_enum_could_be_subclassed():

    class MyEnum(SpecialEnum):

        class Meta:
            other_attr = 'yeah!'
    assert MyEnum._meta.other_attr == 'yeah!'