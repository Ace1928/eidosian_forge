from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
def test_with_contents_from_json_schema(self):
    uri = 'urn:example'
    schema = {'$schema': 'https://json-schema.org/draft/2020-12/schema'}
    registry = Registry().with_contents([(uri, schema)])
    expected = Resource(contents=schema, specification=DRAFT202012)
    assert registry[uri] == expected