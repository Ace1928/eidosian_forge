from tests.unit import unittest
from mock import Mock
from boto.dynamodb.layer2 import Layer2
from boto.dynamodb.table import Table, Schema
def test_schema_not_equal(self):
    s1 = Schema.create(hash_key=('foo', 'N'))
    s2 = Schema.create(hash_key=('bar', 'N'))
    s3 = Schema.create(hash_key=('foo', 'S'))
    self.assertNotEqual(s1, s2)
    self.assertNotEqual(s1, s3)