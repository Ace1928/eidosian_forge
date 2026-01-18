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
def test_objecttype_on_instances():

    class Ship:

        def __init__(self, name):
            self.name = name

    class ShipType(ObjectType):
        name = String(description='Ship name', required=True)

        def resolve_name(self, info):
            return self.name

    class Query(ObjectType):
        ship = Field(ShipType)

        def resolve_ship(self, info):
            return Ship(name='xwing')
    schema = Schema(query=Query)
    executed = schema.execute('{ ship { name } }')
    assert not executed.errors
    assert executed.data == {'ship': {'name': 'xwing'}}