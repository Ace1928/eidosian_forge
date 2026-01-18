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
def test_update_global_secondary_index(self):
    with mock.patch.object(self.users.connection, 'update_table', return_value={}) as mock_update:
        self.users.update_global_secondary_index(global_indexes={'A_IndexToBeUpdated': {'read': 5, 'write': 5}})
    mock_update.assert_called_once_with('users', global_secondary_index_updates=[{'Update': {'IndexName': 'A_IndexToBeUpdated', 'ProvisionedThroughput': {'ReadCapacityUnits': 5, 'WriteCapacityUnits': 5}}}])
    with mock.patch.object(self.users.connection, 'update_table', return_value={}) as mock_update:
        self.users.update_global_secondary_index(global_indexes={'A_IndexToBeUpdated': {'read': 5, 'write': 5}, 'B_IndexToBeUpdated': {'read': 9, 'write': 9}})
    args, kwargs = mock_update.call_args
    self.assertEqual(args, ('users',))
    update = kwargs['global_secondary_index_updates'][:]
    update.sort(key=lambda x: x['Update']['IndexName'])
    self.assertDictEqual(update[0], {'Update': {'IndexName': 'A_IndexToBeUpdated', 'ProvisionedThroughput': {'WriteCapacityUnits': 5, 'ReadCapacityUnits': 5}}})
    self.assertDictEqual(update[1], {'Update': {'IndexName': 'B_IndexToBeUpdated', 'ProvisionedThroughput': {'WriteCapacityUnits': 9, 'ReadCapacityUnits': 9}}})