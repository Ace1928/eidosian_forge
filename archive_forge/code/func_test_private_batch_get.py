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
def test_private_batch_get(self):
    expected = {'ConsumedCapacity': {'CapacityUnits': 0.5, 'TableName': 'users'}, 'Responses': {'users': [{'username': {'S': 'alice'}, 'first_name': {'S': 'Alice'}, 'last_name': {'S': 'Expert'}, 'date_joined': {'N': '1366056680'}, 'friend_count': {'N': '1'}, 'friends': {'SS': ['jane']}}, {'username': {'S': 'bob'}, 'first_name': {'S': 'Bob'}, 'last_name': {'S': 'Smith'}, 'date_joined': {'N': '1366056888'}, 'friend_count': {'N': '1'}, 'friends': {'SS': ['johndoe']}}, {'username': {'S': 'jane'}, 'first_name': {'S': 'Jane'}, 'last_name': {'S': 'Doe'}, 'date_joined': {'N': '1366057777'}, 'friend_count': {'N': '2'}, 'friends': {'SS': ['alice', 'johndoe']}}]}, 'UnprocessedKeys': {}}
    with mock.patch.object(self.users.connection, 'batch_get_item', return_value=expected) as mock_batch_get:
        results = self.users._batch_get(keys=[{'username': 'alice', 'friend_count': 1}, {'username': 'bob', 'friend_count': 1}, {'username': 'jane'}])
        usernames = [res['username'] for res in results['results']]
        self.assertEqual(usernames, ['alice', 'bob', 'jane'])
        self.assertEqual(len(results['results']), 3)
        self.assertEqual(results['last_key'], None)
        self.assertEqual(results['unprocessed_keys'], [])
    mock_batch_get.assert_called_once_with(request_items={'users': {'Keys': [{'username': {'S': 'alice'}, 'friend_count': {'N': '1'}}, {'username': {'S': 'bob'}, 'friend_count': {'N': '1'}}, {'username': {'S': 'jane'}}]}})
    del expected['Responses']['users'][2]
    expected['UnprocessedKeys'] = {'users': {'Keys': [{'username': {'S': 'jane'}}]}}
    with mock.patch.object(self.users.connection, 'batch_get_item', return_value=expected) as mock_batch_get_2:
        results = self.users._batch_get(keys=[{'username': 'alice', 'friend_count': 1}, {'username': 'bob', 'friend_count': 1}, {'username': 'jane'}])
        usernames = [res['username'] for res in results['results']]
        self.assertEqual(usernames, ['alice', 'bob'])
        self.assertEqual(len(results['results']), 2)
        self.assertEqual(results['last_key'], None)
        self.assertEqual(results['unprocessed_keys'], [{'username': 'jane'}])
    mock_batch_get_2.assert_called_once_with(request_items={'users': {'Keys': [{'username': {'S': 'alice'}, 'friend_count': {'N': '1'}}, {'username': {'S': 'bob'}, 'friend_count': {'N': '1'}}, {'username': {'S': 'jane'}}]}})