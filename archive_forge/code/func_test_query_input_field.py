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
def test_query_input_field():

    class Input(InputObjectType):
        a_field = String()
        recursive_field = InputField(lambda: Input)

    class Query(ObjectType):
        test = String(a_input=Input())

        def resolve_test(self, info, **args):
            return json.dumps([self, args], separators=(',', ':'))
    test_schema = Schema(Query)
    result = test_schema.execute('{ test }', None)
    assert not result.errors
    assert result.data == {'test': '[null,{}]'}
    result = test_schema.execute('{ test(aInput: {aField: "String!"} ) }', 'Source!')
    assert not result.errors
    assert result.data == {'test': '["Source!",{"a_input":{"a_field":"String!"}}]'}
    result = test_schema.execute('{ test(aInput: {recursiveField: {aField: "String!"}}) }', 'Source!')
    assert not result.errors
    assert result.data == {'test': '["Source!",{"a_input":{"recursive_field":{"a_field":"String!"}}}]'}