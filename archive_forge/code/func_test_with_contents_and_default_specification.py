from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
def test_with_contents_and_default_specification(self):
    uri = 'urn:example'
    registry = Registry().with_contents([(uri, {'foo': 'bar'})], default_specification=Specification.OPAQUE)
    assert registry[uri] == Resource.opaque({'foo': 'bar'})