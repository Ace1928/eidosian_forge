from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
def test_retrieve_arbitrary_exception(self):
    foo = Resource.opaque({'foo': 'bar'})

    def retrieve(uri):
        if uri == 'urn:succeed':
            return foo
        raise Exception('Oh no!')
    registry = Registry(retrieve=retrieve)
    assert registry.get_or_retrieve('urn:succeed').value == foo
    with pytest.raises(exceptions.Unretrievable):
        registry.get_or_retrieve('urn:uhoh')