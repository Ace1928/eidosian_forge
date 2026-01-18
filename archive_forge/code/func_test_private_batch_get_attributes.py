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
def test_private_batch_get_attributes(self):
    expected = {'ConsumedCapacity': {'CapacityUnits': 0.5, 'TableName': 'users'}, 'Responses': {'users': [{'username': {'S': 'alice'}, 'first_name': {'S': 'Alice'}}, {'username': {'S': 'bob'}, 'first_name': {'S': 'Bob'}}]}, 'UnprocessedKeys': {}}
    with mock.patch.object(self.users.connection, 'batch_get_item', return_value=expected) as mock_batch_get_attr:
        results = self.users._batch_get(keys=[{'username': 'alice'}, {'username': 'bob'}], attributes=['username', 'first_name'])
        usernames = [res['username'] for res in results['results']]
        first_names = [res['first_name'] for res in results['results']]
        self.assertEqual(usernames, ['alice', 'bob'])
        self.assertEqual(first_names, ['Alice', 'Bob'])
        self.assertEqual(len(results['results']), 2)
        self.assertEqual(results['last_key'], None)
        self.assertEqual(results['unprocessed_keys'], [])
    mock_batch_get_attr.assert_called_once_with(request_items={'users': {'Keys': [{'username': {'S': 'alice'}}, {'username': {'S': 'bob'}}], 'AttributesToGet': ['username', 'first_name']}})