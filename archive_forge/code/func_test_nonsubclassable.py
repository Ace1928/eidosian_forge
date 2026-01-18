from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
@pytest.mark.parametrize('cls', [Anchor, Registry, Resource, Specification, exceptions.PointerToNowhere])
def test_nonsubclassable(cls):
    with pytest.raises(Exception, match='(?i)subclassing'):

        class Boom(cls):
            pass