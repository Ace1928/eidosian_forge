import copy
from ..argument import Argument
from ..definitions import GrapheneGraphQLType
from ..enum import Enum
from ..field import Field
from ..inputfield import InputField
from ..inputobjecttype import InputObjectType
from ..interface import Interface
from ..objecttype import ObjectType
from ..scalars import Boolean, Int, String
from ..schema import Schema
from ..structures import List, NonNull
from ..union import Union
def test_stringifies_simple_types():
    assert str(Int) == 'Int'
    assert str(Article) == 'Article'
    assert str(MyInterface) == 'MyInterface'
    assert str(MyUnion) == 'MyUnion'
    assert str(MyEnum) == 'MyEnum'
    assert str(MyInputObjectType) == 'MyInputObjectType'
    assert str(NonNull(Int)) == 'Int!'
    assert str(List(Int)) == '[Int]'
    assert str(NonNull(List(Int))) == '[Int]!'
    assert str(List(NonNull(Int))) == '[Int!]'
    assert str(List(List(Int))) == '[[Int]]'