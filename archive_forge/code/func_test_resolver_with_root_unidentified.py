from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
def test_resolver_with_root_unidentified(self):
    root = Resource.opaque(contents={})
    resolver = Registry().resolver_with_root(root)
    assert resolver.lookup('#').contents == root.contents