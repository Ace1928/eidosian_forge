from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
def test_repeated_anchor_lookup_from_retrieved_resource(self):
    resource = Resource.opaque(contents={'foo': 'baz'})
    once = [resource]

    def retrieve(uri):
        return once.pop()
    resolver = Registry(retrieve=retrieve).resolver()
    resolved = resolver.lookup('http://example.com/')
    assert resolved.contents == resource.contents
    resolved = resolved.resolver.lookup('#')
    assert resolved.contents == resource.contents