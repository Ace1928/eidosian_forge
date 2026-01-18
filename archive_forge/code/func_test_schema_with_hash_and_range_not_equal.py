from tests.unit import unittest
from mock import Mock
from boto.dynamodb.layer2 import Layer2
from boto.dynamodb.table import Table, Schema
def test_schema_with_hash_and_range_not_equal(self):
    s1 = Schema.create(hash_key=('foo', 'N'), range_key=('bar', 'S'))
    s2 = Schema.create(hash_key=('foo', 'N'), range_key=('bar', 'N'))
    s3 = Schema.create(hash_key=('foo', 'S'), range_key=('baz', 'N'))
    s4 = Schema.create(hash_key=('bar', 'N'), range_key=('baz', 'N'))
    self.assertNotEqual(s1, s2)
    self.assertNotEqual(s1, s3)
    self.assertNotEqual(s1, s4)
    self.assertNotEqual(s2, s4)
    self.assertNotEqual(s3, s4)