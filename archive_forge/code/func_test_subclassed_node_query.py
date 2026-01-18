import re
from textwrap import dedent
from graphql_relay import to_global_id
from ...types import ObjectType, Schema, String
from ..node import Node, is_node
def test_subclassed_node_query():
    executed = schema.execute('{ node(id:"%s") { ... on MyOtherNode { shared, extraField, somethingElse } } }' % to_global_id('MyOtherNode', 1))
    assert not executed.errors
    assert executed.data == {'node': {'shared': '1', 'extraField': 'extra field info.', 'somethingElse': '----'}}