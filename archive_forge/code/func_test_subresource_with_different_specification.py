from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
def test_subresource_with_different_specification(self):
    schema = {'$schema': 'https://json-schema.org/draft/2020-12/schema'}
    resource = ID_AND_CHILDREN.create_resource({'children': [schema]})
    assert list(resource.subresources()) == [DRAFT202012.create_resource(schema)]