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
def test_private_scan(self):
    expected = {'ConsumedCapacity': {'CapacityUnits': 0.5, 'TableName': 'users'}, 'Count': 4, 'Items': [{'username': {'S': 'alice'}, 'first_name': {'S': 'Alice'}, 'last_name': {'S': 'Expert'}, 'date_joined': {'N': '1366056680'}, 'friend_count': {'N': '1'}, 'friends': {'SS': ['jane']}}, {'username': {'S': 'bob'}, 'first_name': {'S': 'Bob'}, 'last_name': {'S': 'Smith'}, 'date_joined': {'N': '1366056888'}, 'friend_count': {'N': '1'}, 'friends': {'SS': ['johndoe']}}, {'username': {'S': 'jane'}, 'first_name': {'S': 'Jane'}, 'last_name': {'S': 'Doe'}, 'date_joined': {'N': '1366057777'}, 'friend_count': {'N': '2'}, 'friends': {'SS': ['alice', 'johndoe']}}], 'ScannedCount': 4}
    with mock.patch.object(self.users.connection, 'scan', return_value=expected) as mock_scan:
        results = self.users._scan(limit=2, friend_count__lte=2)
        usernames = [res['username'] for res in results['results']]
        self.assertEqual(usernames, ['alice', 'bob', 'jane'])
        self.assertEqual(len(results['results']), 3)
        self.assertEqual(results['last_key'], None)
    mock_scan.assert_called_once_with('users', scan_filter={'friend_count': {'AttributeValueList': [{'N': '2'}], 'ComparisonOperator': 'LE'}}, limit=2, segment=None, attributes_to_get=None, total_segments=None, conditional_operator=None)
    expected['LastEvaluatedKey'] = {'username': {'S': 'jane'}}
    with mock.patch.object(self.users.connection, 'scan', return_value=expected) as mock_scan_2:
        results = self.users._scan(limit=3, friend_count__lte=2, exclusive_start_key={'username': 'adam'}, segment=None, total_segments=None)
        usernames = [res['username'] for res in results['results']]
        self.assertEqual(usernames, ['alice', 'bob', 'jane'])
        self.assertEqual(len(results['results']), 3)
        self.assertEqual(results['last_key'], {'username': 'jane'})
    mock_scan_2.assert_called_once_with('users', scan_filter={'friend_count': {'AttributeValueList': [{'N': '2'}], 'ComparisonOperator': 'LE'}}, limit=3, exclusive_start_key={'username': {'S': 'adam'}}, segment=None, attributes_to_get=None, total_segments=None, conditional_operator=None)