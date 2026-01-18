from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
def test_retrieve_cannot_determine_specification(self):

    def retrieve(uri):
        return Resource.from_contents({})
    registry = Registry(retrieve=retrieve)
    with pytest.raises(exceptions.CannotDetermineSpecification):
        registry.get_or_retrieve('urn:uhoh')