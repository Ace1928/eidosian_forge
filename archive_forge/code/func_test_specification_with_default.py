import pytest
from referencing import Registry, Resource, Specification
import referencing.jsonschema
def test_specification_with_default():
    dialect_id = 'http://example.com/unknown-json-schema-dialect-id'
    specification = referencing.jsonschema.specification_with(dialect_id, default=Specification.OPAQUE)
    assert specification is Specification.OPAQUE