from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
def test_retrieve_no_such_resource(self):
    foo = Resource.opaque({'foo': 'bar'})

    def retrieve(uri):
        if uri == 'urn:succeed':
            return foo
        raise exceptions.NoSuchResource(ref=uri)
    registry = Registry(retrieve=retrieve)
    assert registry.get_or_retrieve('urn:succeed').value == foo
    with pytest.raises(exceptions.NoSuchResource):
        registry.get_or_retrieve('urn:uhoh')