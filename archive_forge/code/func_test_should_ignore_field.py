import re
from pytest import raises
from graphql import parse, get_introspection_query, validate
from ...types import Schema, ObjectType, Interface
from ...types import String, Int, List, Field
from ..depth_limit import depth_limit_validator
def test_should_ignore_field():
    query = '\n    query read1 {\n      user { address { city } }\n    }\n    query read2 {\n      user1 { address { city } }\n      user2 { address { city } }\n      user3 { address { city } }\n    }\n    '
    errors, result = run_query(query, 10, ignore=['user1', re.compile('user2'), lambda field_name: field_name == 'user3'])
    expected = {'read1': 2, 'read2': 0}
    assert not errors
    assert result == expected