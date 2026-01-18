from rpds import HashTrieMap
import pytest
from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012
def test_retrieve_first_checks_crawlable_resource(self):
    child = ID_AND_CHILDREN.create_resource({'ID': 'urn:child', 'foo': 12})
    root = ID_AND_CHILDREN.create_resource({'children': [child.contents]})
    registry = Registry(retrieve=blow_up).with_resource('urn:root', root)
    assert registry.crawl()['urn:child'] == child