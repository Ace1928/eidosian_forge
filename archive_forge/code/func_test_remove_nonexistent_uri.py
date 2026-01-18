from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
def test_remove_nonexistent_uri(self):
    with pytest.raises(exceptions.NoSuchResource) as e:
        Registry().remove('urn:doesNotExist')
    assert e.value == exceptions.NoSuchResource(ref='urn:doesNotExist')