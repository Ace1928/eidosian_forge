import re
from pytest import raises
from graphql import parse, get_introspection_query, validate
from ...types import Schema, ObjectType, Interface
from ...types import String, Int, List, Field
from ..depth_limit import depth_limit_validator
def test_should_count_depth_without_fragment():
    query = '\n    query read0 {\n      version\n    }\n    query read1 {\n      version\n      user {\n        name\n      }\n    }\n    query read2 {\n      matt: user(name: "matt") {\n        email\n      }\n      andy: user(name: "andy") {\n        email\n        address {\n          city\n        }\n      }\n    }\n    query read3 {\n      matt: user(name: "matt") {\n        email\n      }\n      andy: user(name: "andy") {\n        email\n        address {\n          city\n        }\n        pets {\n          name\n          owner {\n            name\n          }\n        }\n      }\n    }\n    '
    expected = {'read0': 0, 'read1': 1, 'read2': 2, 'read3': 3}
    errors, result = run_query(query, 10)
    assert not errors
    assert result == expected