from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
def test_resolver(self):
    one = Resource.opaque(contents={})
    registry = Registry({'http://example.com': one})
    resolver = registry.resolver(base_uri='http://example.com')
    assert resolver.lookup('#').contents == {}