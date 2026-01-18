import re
from uuid import uuid4
from graphql import graphql_sync
from ..id_type import BaseGlobalIDType, SimpleGlobalIDType, UUIDGlobalIDType
from ..node import Node
from ...types import Int, ObjectType, Schema, String
def test_str_schema_correct(self):
    """
        Check that the schema has the expected and custom node interface and user type and that they both use UUIDs
        """
    parsed = re.findall('(.+) \\{\\n\\s*([\\w\\W]*?)\\n\\}', str(self.schema))
    types = [t for t, f in parsed]
    fields = [f for t, f in parsed]
    custom_node_interface = 'interface CustomNode'
    assert custom_node_interface in types
    assert '"""The ID of the object"""\n  id: Int!' == fields[types.index(custom_node_interface)]
    user_type = 'type User implements CustomNode'
    assert user_type in types
    assert '"""The ID of the object"""\n  id: Int!\n  name: String' == fields[types.index(user_type)]