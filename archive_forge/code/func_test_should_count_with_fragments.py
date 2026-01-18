import re
from pytest import raises
from graphql import parse, get_introspection_query, validate
from ...types import Schema, ObjectType, Interface
from ...types import String, Int, List, Field
from ..depth_limit import depth_limit_validator
def test_should_count_with_fragments():
    query = '\n    query read0 {\n      ... on Query {\n        version\n      }\n    }\n    query read1 {\n      version\n      user {\n        ... on Human {\n          name\n        }\n      }\n    }\n    fragment humanInfo on Human {\n      email\n    }\n    fragment petInfo on Pet {\n      name\n      owner {\n        name\n      }\n    }\n    query read2 {\n      matt: user(name: "matt") {\n        ...humanInfo\n      }\n      andy: user(name: "andy") {\n        ...humanInfo\n        address {\n          city\n        }\n      }\n    }\n    query read3 {\n      matt: user(name: "matt") {\n        ...humanInfo\n      }\n      andy: user(name: "andy") {\n        ... on Human {\n          email\n        }\n        address {\n          city\n        }\n        pets {\n          ...petInfo\n        }\n      }\n    }\n  '
    expected = {'read0': 0, 'read1': 1, 'read2': 2, 'read3': 3}
    errors, result = run_query(query, 10)
    assert not errors
    assert result == expected