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
def test_query_middlewares():

    class Query(ObjectType):
        hello = String()
        other = String()

        def resolve_hello(self, info):
            return 'World'

        def resolve_other(self, info):
            return 'other'

    def reversed_middleware(next, *args, **kwargs):
        return next(*args, **kwargs)[::-1]
    hello_schema = Schema(Query)
    executed = hello_schema.execute('{ hello, other }', middleware=[reversed_middleware])
    assert not executed.errors
    assert executed.data == {'hello': 'dlroW', 'other': 'rehto'}