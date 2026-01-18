from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
def test_with_resources(self):
    """
        Adding multiple resources to the registry is like adding each one.
        """
    one = Resource.opaque(contents={})
    two = Resource(contents={'foo': 'bar'}, specification=ID_AND_CHILDREN)
    registry = Registry().with_resources([('http://example.com/1', one), ('http://example.com/foo/bar', two)])
    assert registry == Registry().with_resource(uri='http://example.com/1', resource=one).with_resource(uri='http://example.com/foo/bar', resource=two)