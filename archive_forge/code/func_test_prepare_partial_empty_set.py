from tests.compat import mock, unittest
from boto.dynamodb2 import exceptions
from boto.dynamodb2.fields import (HashKey, RangeKey,
from boto.dynamodb2.items import Item
from boto.dynamodb2.layer1 import DynamoDBConnection
from boto.dynamodb2.results import ResultSet, BatchGetResultSet
from boto.dynamodb2.table import Table
from boto.dynamodb2.types import (STRING, NUMBER, BINARY,
from boto.exception import JSONResponseError
from boto.compat import six, long_type
def test_prepare_partial_empty_set(self):
    self.johndoe.mark_clean()
    self.johndoe['first_name'] = 'Johann'
    self.johndoe['last_name'] = 'Doe'
    del self.johndoe['date_joined']
    self.johndoe['friends'] = set()
    final_data, fields = self.johndoe.prepare_partial()
    self.assertEqual(final_data, {'date_joined': {'Action': 'DELETE'}, 'first_name': {'Action': 'PUT', 'Value': {'S': 'Johann'}}, 'last_name': {'Action': 'PUT', 'Value': {'S': 'Doe'}}})
    self.assertEqual(fields, set(['first_name', 'last_name', 'date_joined']))