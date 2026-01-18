from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
def test_retrieve(self):
    foo = Resource.opaque({'foo': 'bar'})
    registry = Registry(retrieve=lambda uri: foo)
    assert registry.get_or_retrieve('urn:example').value == foo