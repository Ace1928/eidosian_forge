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
class IndexFieldTestCase(unittest.TestCase):

    def test_all_index(self):
        all_index = AllIndex('AllKeys', parts=[HashKey('username'), RangeKey('date_joined')])
        self.assertEqual(all_index.name, 'AllKeys')
        self.assertEqual([part.attr_type for part in all_index.parts], ['HASH', 'RANGE'])
        self.assertEqual(all_index.projection_type, 'ALL')
        self.assertEqual(all_index.definition(), [{'AttributeName': 'username', 'AttributeType': 'S'}, {'AttributeName': 'date_joined', 'AttributeType': 'S'}])
        self.assertEqual(all_index.schema(), {'IndexName': 'AllKeys', 'KeySchema': [{'AttributeName': 'username', 'KeyType': 'HASH'}, {'AttributeName': 'date_joined', 'KeyType': 'RANGE'}], 'Projection': {'ProjectionType': 'ALL'}})

    def test_keys_only_index(self):
        keys_only = KeysOnlyIndex('KeysOnly', parts=[HashKey('username'), RangeKey('date_joined')])
        self.assertEqual(keys_only.name, 'KeysOnly')
        self.assertEqual([part.attr_type for part in keys_only.parts], ['HASH', 'RANGE'])
        self.assertEqual(keys_only.projection_type, 'KEYS_ONLY')
        self.assertEqual(keys_only.definition(), [{'AttributeName': 'username', 'AttributeType': 'S'}, {'AttributeName': 'date_joined', 'AttributeType': 'S'}])
        self.assertEqual(keys_only.schema(), {'IndexName': 'KeysOnly', 'KeySchema': [{'AttributeName': 'username', 'KeyType': 'HASH'}, {'AttributeName': 'date_joined', 'KeyType': 'RANGE'}], 'Projection': {'ProjectionType': 'KEYS_ONLY'}})

    def test_include_index(self):
        include_index = IncludeIndex('IncludeKeys', parts=[HashKey('username'), RangeKey('date_joined')], includes=['gender', 'friend_count'])
        self.assertEqual(include_index.name, 'IncludeKeys')
        self.assertEqual([part.attr_type for part in include_index.parts], ['HASH', 'RANGE'])
        self.assertEqual(include_index.projection_type, 'INCLUDE')
        self.assertEqual(include_index.definition(), [{'AttributeName': 'username', 'AttributeType': 'S'}, {'AttributeName': 'date_joined', 'AttributeType': 'S'}])
        self.assertEqual(include_index.schema(), {'IndexName': 'IncludeKeys', 'KeySchema': [{'AttributeName': 'username', 'KeyType': 'HASH'}, {'AttributeName': 'date_joined', 'KeyType': 'RANGE'}], 'Projection': {'ProjectionType': 'INCLUDE', 'NonKeyAttributes': ['gender', 'friend_count']}})

    def test_global_all_index(self):
        all_index = GlobalAllIndex('AllKeys', parts=[HashKey('username'), RangeKey('date_joined')], throughput={'read': 6, 'write': 2})
        self.assertEqual(all_index.name, 'AllKeys')
        self.assertEqual([part.attr_type for part in all_index.parts], ['HASH', 'RANGE'])
        self.assertEqual(all_index.projection_type, 'ALL')
        self.assertEqual(all_index.definition(), [{'AttributeName': 'username', 'AttributeType': 'S'}, {'AttributeName': 'date_joined', 'AttributeType': 'S'}])
        self.assertEqual(all_index.schema(), {'IndexName': 'AllKeys', 'KeySchema': [{'AttributeName': 'username', 'KeyType': 'HASH'}, {'AttributeName': 'date_joined', 'KeyType': 'RANGE'}], 'Projection': {'ProjectionType': 'ALL'}, 'ProvisionedThroughput': {'ReadCapacityUnits': 6, 'WriteCapacityUnits': 2}})

    def test_global_keys_only_index(self):
        keys_only = GlobalKeysOnlyIndex('KeysOnly', parts=[HashKey('username'), RangeKey('date_joined')], throughput={'read': 3, 'write': 4})
        self.assertEqual(keys_only.name, 'KeysOnly')
        self.assertEqual([part.attr_type for part in keys_only.parts], ['HASH', 'RANGE'])
        self.assertEqual(keys_only.projection_type, 'KEYS_ONLY')
        self.assertEqual(keys_only.definition(), [{'AttributeName': 'username', 'AttributeType': 'S'}, {'AttributeName': 'date_joined', 'AttributeType': 'S'}])
        self.assertEqual(keys_only.schema(), {'IndexName': 'KeysOnly', 'KeySchema': [{'AttributeName': 'username', 'KeyType': 'HASH'}, {'AttributeName': 'date_joined', 'KeyType': 'RANGE'}], 'Projection': {'ProjectionType': 'KEYS_ONLY'}, 'ProvisionedThroughput': {'ReadCapacityUnits': 3, 'WriteCapacityUnits': 4}})

    def test_global_include_index(self):
        include_index = GlobalIncludeIndex('IncludeKeys', parts=[HashKey('username'), RangeKey('date_joined')], includes=['gender', 'friend_count'])
        self.assertEqual(include_index.name, 'IncludeKeys')
        self.assertEqual([part.attr_type for part in include_index.parts], ['HASH', 'RANGE'])
        self.assertEqual(include_index.projection_type, 'INCLUDE')
        self.assertEqual(include_index.definition(), [{'AttributeName': 'username', 'AttributeType': 'S'}, {'AttributeName': 'date_joined', 'AttributeType': 'S'}])
        self.assertEqual(include_index.schema(), {'IndexName': 'IncludeKeys', 'KeySchema': [{'AttributeName': 'username', 'KeyType': 'HASH'}, {'AttributeName': 'date_joined', 'KeyType': 'RANGE'}], 'Projection': {'ProjectionType': 'INCLUDE', 'NonKeyAttributes': ['gender', 'friend_count']}, 'ProvisionedThroughput': {'ReadCapacityUnits': 5, 'WriteCapacityUnits': 5}})

    def test_global_include_index_throughput(self):
        include_index = GlobalIncludeIndex('IncludeKeys', parts=[HashKey('username'), RangeKey('date_joined')], includes=['gender', 'friend_count'], throughput={'read': 10, 'write': 8})
        self.assertEqual(include_index.schema(), {'IndexName': 'IncludeKeys', 'KeySchema': [{'AttributeName': 'username', 'KeyType': 'HASH'}, {'AttributeName': 'date_joined', 'KeyType': 'RANGE'}], 'Projection': {'ProjectionType': 'INCLUDE', 'NonKeyAttributes': ['gender', 'friend_count']}, 'ProvisionedThroughput': {'ReadCapacityUnits': 10, 'WriteCapacityUnits': 8}})