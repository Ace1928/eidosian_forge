import json
from functools import partial
from graphql import (
from ..context import Context
from ..dynamic import Dynamic
from ..field import Field
from ..inputfield import InputField
from ..inputobjecttype import InputObjectType
from ..interface import Interface
from ..objecttype import ObjectType
from ..scalars import Boolean, Int, String
from ..schema import Schema
from ..structures import List, NonNull
from ..union import Union
def test_query_interface():

    class one_object:
        pass

    class two_object:
        pass

    class MyInterface(Interface):
        base = String()

    class One(ObjectType):

        class Meta:
            interfaces = (MyInterface,)
        one = String()

        @classmethod
        def is_type_of(cls, root, info):
            return isinstance(root, one_object)

    class Two(ObjectType):

        class Meta:
            interfaces = (MyInterface,)
        two = String()

        @classmethod
        def is_type_of(cls, root, info):
            return isinstance(root, two_object)

    class Query(ObjectType):
        interfaces = List(MyInterface)

        def resolve_interfaces(self, info):
            return [one_object(), two_object()]
    hello_schema = Schema(Query, types=[One, Two])
    executed = hello_schema.execute('{ interfaces { __typename } }')
    assert not executed.errors
    assert executed.data == {'interfaces': [{'__typename': 'One'}, {'__typename': 'Two'}]}