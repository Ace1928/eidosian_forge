import time
import uuid
from decimal import Decimal
from tests.unit import unittest
from boto.dynamodb.exceptions import DynamoDBKeyNotFoundError
from boto.dynamodb.exceptions import DynamoDBConditionalCheckFailedError
from boto.dynamodb.layer2 import Layer2
from boto.dynamodb.types import get_dynamodb_type, Binary
from boto.dynamodb.condition import BEGINS_WITH, CONTAINS, GT
from boto.compat import six, long_type
def test_put_single_letter_attr(self):
    table = self.create_sample_table()
    item = table.new_item('foo', 'foo1')
    item.put_attribute('b', 4)
    stored = item.save(return_values='UPDATED_NEW')
    self.assertEqual(stored['Attributes'], {'b': 4})