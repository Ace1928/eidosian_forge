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
class TableTestCase(unittest.TestCase):

    def setUp(self):
        super(TableTestCase, self).setUp()
        self.users = Table('users', connection=FakeDynamoDBConnection())
        self.default_connection = DynamoDBConnection(aws_access_key_id='access_key', aws_secret_access_key='secret_key')

    def test__introspect_schema(self):
        raw_schema_1 = [{'AttributeName': 'username', 'KeyType': 'HASH'}, {'AttributeName': 'date_joined', 'KeyType': 'RANGE'}]
        raw_attributes_1 = [{'AttributeName': 'username', 'AttributeType': 'S'}, {'AttributeName': 'date_joined', 'AttributeType': 'S'}]
        schema_1 = self.users._introspect_schema(raw_schema_1, raw_attributes_1)
        self.assertEqual(len(schema_1), 2)
        self.assertTrue(isinstance(schema_1[0], HashKey))
        self.assertEqual(schema_1[0].name, 'username')
        self.assertTrue(isinstance(schema_1[1], RangeKey))
        self.assertEqual(schema_1[1].name, 'date_joined')
        raw_schema_2 = [{'AttributeName': 'username', 'KeyType': 'BTREE'}]
        raw_attributes_2 = [{'AttributeName': 'username', 'AttributeType': 'S'}]
        self.assertRaises(exceptions.UnknownSchemaFieldError, self.users._introspect_schema, raw_schema_2, raw_attributes_2)
        raw_schema_3 = [{'AttributeName': 'user_id', 'KeyType': 'HASH'}, {'AttributeName': 'junk', 'KeyType': 'RANGE'}]
        raw_attributes_3 = [{'AttributeName': 'user_id', 'AttributeType': 'N'}, {'AttributeName': 'junk', 'AttributeType': 'B'}]
        schema_3 = self.users._introspect_schema(raw_schema_3, raw_attributes_3)
        self.assertEqual(len(schema_3), 2)
        self.assertTrue(isinstance(schema_3[0], HashKey))
        self.assertEqual(schema_3[0].name, 'user_id')
        self.assertEqual(schema_3[0].data_type, NUMBER)
        self.assertTrue(isinstance(schema_3[1], RangeKey))
        self.assertEqual(schema_3[1].name, 'junk')
        self.assertEqual(schema_3[1].data_type, BINARY)

    def test__introspect_indexes(self):
        raw_indexes_1 = [{'IndexName': 'MostRecentlyJoinedIndex', 'KeySchema': [{'AttributeName': 'username', 'KeyType': 'HASH'}, {'AttributeName': 'date_joined', 'KeyType': 'RANGE'}], 'Projection': {'ProjectionType': 'KEYS_ONLY'}}, {'IndexName': 'EverybodyIndex', 'KeySchema': [{'AttributeName': 'username', 'KeyType': 'HASH'}], 'Projection': {'ProjectionType': 'ALL'}}, {'IndexName': 'GenderIndex', 'KeySchema': [{'AttributeName': 'username', 'KeyType': 'HASH'}, {'AttributeName': 'date_joined', 'KeyType': 'RANGE'}], 'Projection': {'ProjectionType': 'INCLUDE', 'NonKeyAttributes': ['gender']}}]
        indexes_1 = self.users._introspect_indexes(raw_indexes_1)
        self.assertEqual(len(indexes_1), 3)
        self.assertTrue(isinstance(indexes_1[0], KeysOnlyIndex))
        self.assertEqual(indexes_1[0].name, 'MostRecentlyJoinedIndex')
        self.assertEqual(len(indexes_1[0].parts), 2)
        self.assertTrue(isinstance(indexes_1[1], AllIndex))
        self.assertEqual(indexes_1[1].name, 'EverybodyIndex')
        self.assertEqual(len(indexes_1[1].parts), 1)
        self.assertTrue(isinstance(indexes_1[2], IncludeIndex))
        self.assertEqual(indexes_1[2].name, 'GenderIndex')
        self.assertEqual(len(indexes_1[2].parts), 2)
        self.assertEqual(indexes_1[2].includes_fields, ['gender'])
        raw_indexes_2 = [{'IndexName': 'MostRecentlyJoinedIndex', 'KeySchema': [{'AttributeName': 'username', 'KeyType': 'HASH'}, {'AttributeName': 'date_joined', 'KeyType': 'RANGE'}], 'Projection': {'ProjectionType': 'SOMETHING_CRAZY'}}]
        self.assertRaises(exceptions.UnknownIndexFieldError, self.users._introspect_indexes, raw_indexes_2)

    def test_initialization(self):
        users = Table('users', connection=self.default_connection)
        self.assertEqual(users.table_name, 'users')
        self.assertTrue(isinstance(users.connection, DynamoDBConnection))
        self.assertEqual(users.throughput['read'], 5)
        self.assertEqual(users.throughput['write'], 5)
        self.assertEqual(users.schema, None)
        self.assertEqual(users.indexes, None)
        groups = Table('groups', connection=FakeDynamoDBConnection())
        self.assertEqual(groups.table_name, 'groups')
        self.assertTrue(hasattr(groups.connection, 'assert_called_once_with'))

    def test_create_simple(self):
        conn = FakeDynamoDBConnection()
        with mock.patch.object(conn, 'create_table', return_value={}) as mock_create_table:
            retval = Table.create('users', schema=[HashKey('username'), RangeKey('date_joined', data_type=NUMBER)], connection=conn)
            self.assertTrue(retval)
        self.assertTrue(mock_create_table.called)
        mock_create_table.assert_called_once_with(attribute_definitions=[{'AttributeName': 'username', 'AttributeType': 'S'}, {'AttributeName': 'date_joined', 'AttributeType': 'N'}], table_name='users', key_schema=[{'KeyType': 'HASH', 'AttributeName': 'username'}, {'KeyType': 'RANGE', 'AttributeName': 'date_joined'}], provisioned_throughput={'WriteCapacityUnits': 5, 'ReadCapacityUnits': 5})

    def test_create_full(self):
        conn = FakeDynamoDBConnection()
        with mock.patch.object(conn, 'create_table', return_value={}) as mock_create_table:
            retval = Table.create('users', schema=[HashKey('username'), RangeKey('date_joined', data_type=NUMBER)], throughput={'read': 20, 'write': 10}, indexes=[KeysOnlyIndex('FriendCountIndex', parts=[RangeKey('friend_count')])], global_indexes=[GlobalKeysOnlyIndex('FullFriendCountIndex', parts=[RangeKey('friend_count')], throughput={'read': 10, 'write': 8})], connection=conn)
            self.assertTrue(retval)
        self.assertTrue(mock_create_table.called)
        mock_create_table.assert_called_once_with(attribute_definitions=[{'AttributeName': 'username', 'AttributeType': 'S'}, {'AttributeName': 'date_joined', 'AttributeType': 'N'}, {'AttributeName': 'friend_count', 'AttributeType': 'S'}], key_schema=[{'KeyType': 'HASH', 'AttributeName': 'username'}, {'KeyType': 'RANGE', 'AttributeName': 'date_joined'}], table_name='users', provisioned_throughput={'WriteCapacityUnits': 10, 'ReadCapacityUnits': 20}, global_secondary_indexes=[{'KeySchema': [{'KeyType': 'RANGE', 'AttributeName': 'friend_count'}], 'IndexName': 'FullFriendCountIndex', 'Projection': {'ProjectionType': 'KEYS_ONLY'}, 'ProvisionedThroughput': {'WriteCapacityUnits': 8, 'ReadCapacityUnits': 10}}], local_secondary_indexes=[{'KeySchema': [{'KeyType': 'RANGE', 'AttributeName': 'friend_count'}], 'IndexName': 'FriendCountIndex', 'Projection': {'ProjectionType': 'KEYS_ONLY'}}])

    def test_describe(self):
        expected = {'Table': {'AttributeDefinitions': [{'AttributeName': 'username', 'AttributeType': 'S'}], 'ItemCount': 5, 'KeySchema': [{'AttributeName': 'username', 'KeyType': 'HASH'}], 'LocalSecondaryIndexes': [{'IndexName': 'UsernameIndex', 'KeySchema': [{'AttributeName': 'username', 'KeyType': 'HASH'}], 'Projection': {'ProjectionType': 'KEYS_ONLY'}}], 'ProvisionedThroughput': {'ReadCapacityUnits': 20, 'WriteCapacityUnits': 6}, 'TableName': 'Thread', 'TableStatus': 'ACTIVE'}}
        with mock.patch.object(self.users.connection, 'describe_table', return_value=expected) as mock_describe:
            self.assertEqual(self.users.throughput['read'], 5)
            self.assertEqual(self.users.throughput['write'], 5)
            self.assertEqual(self.users.schema, None)
            self.assertEqual(self.users.indexes, None)
            self.users.describe()
            self.assertEqual(self.users.throughput['read'], 20)
            self.assertEqual(self.users.throughput['write'], 6)
            self.assertEqual(len(self.users.schema), 1)
            self.assertEqual(isinstance(self.users.schema[0], HashKey), 1)
            self.assertEqual(len(self.users.indexes), 1)
        mock_describe.assert_called_once_with('users')

    def test_update(self):
        with mock.patch.object(self.users.connection, 'update_table', return_value={}) as mock_update:
            self.assertEqual(self.users.throughput['read'], 5)
            self.assertEqual(self.users.throughput['write'], 5)
            self.users.update(throughput={'read': 7, 'write': 2})
            self.assertEqual(self.users.throughput['read'], 7)
            self.assertEqual(self.users.throughput['write'], 2)
        mock_update.assert_called_once_with('users', global_secondary_index_updates=None, provisioned_throughput={'WriteCapacityUnits': 2, 'ReadCapacityUnits': 7})
        with mock.patch.object(self.users.connection, 'update_table', return_value={}) as mock_update:
            self.assertEqual(self.users.throughput['read'], 7)
            self.assertEqual(self.users.throughput['write'], 2)
            self.users.update(throughput={'read': 9, 'write': 5}, global_indexes={'WhateverIndex': {'read': 6, 'write': 1}, 'AnotherIndex': {'read': 1, 'write': 2}})
            self.assertEqual(self.users.throughput['read'], 9)
            self.assertEqual(self.users.throughput['write'], 5)
        args, kwargs = mock_update.call_args
        self.assertEqual(args, ('users',))
        self.assertEqual(kwargs['provisioned_throughput'], {'WriteCapacityUnits': 5, 'ReadCapacityUnits': 9})
        update = kwargs['global_secondary_index_updates'][:]
        update.sort(key=lambda x: x['Update']['IndexName'])
        self.assertDictEqual(update[0], {'Update': {'IndexName': 'AnotherIndex', 'ProvisionedThroughput': {'WriteCapacityUnits': 2, 'ReadCapacityUnits': 1}}})
        self.assertDictEqual(update[1], {'Update': {'IndexName': 'WhateverIndex', 'ProvisionedThroughput': {'WriteCapacityUnits': 1, 'ReadCapacityUnits': 6}}})

    def test_create_global_secondary_index(self):
        with mock.patch.object(self.users.connection, 'update_table', return_value={}) as mock_update:
            self.users.create_global_secondary_index(global_index=GlobalAllIndex('JustCreatedIndex', parts=[HashKey('requiredHashKey')], throughput={'read': 2, 'write': 2}))
        mock_update.assert_called_once_with('users', global_secondary_index_updates=[{'Create': {'IndexName': 'JustCreatedIndex', 'KeySchema': [{'KeyType': 'HASH', 'AttributeName': 'requiredHashKey'}], 'Projection': {'ProjectionType': 'ALL'}, 'ProvisionedThroughput': {'WriteCapacityUnits': 2, 'ReadCapacityUnits': 2}}}], attribute_definitions=[{'AttributeName': 'requiredHashKey', 'AttributeType': 'S'}])

    def test_delete_global_secondary_index(self):
        with mock.patch.object(self.users.connection, 'update_table', return_value={}) as mock_update:
            self.users.delete_global_secondary_index('RandomGSIIndex')
        mock_update.assert_called_once_with('users', global_secondary_index_updates=[{'Delete': {'IndexName': 'RandomGSIIndex'}}])

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

    def test_delete(self):
        with mock.patch.object(self.users.connection, 'delete_table', return_value={}) as mock_delete:
            self.assertTrue(self.users.delete())
        mock_delete.assert_called_once_with('users')

    def test_get_item(self):
        expected = {'Item': {'username': {'S': 'johndoe'}, 'first_name': {'S': 'John'}, 'last_name': {'S': 'Doe'}, 'date_joined': {'N': '1366056668'}, 'friend_count': {'N': '3'}, 'friends': {'SS': ['alice', 'bob', 'jane']}}}
        with mock.patch.object(self.users.connection, 'get_item', return_value=expected) as mock_get_item:
            item = self.users.get_item(username='johndoe')
            self.assertEqual(item['username'], 'johndoe')
            self.assertEqual(item['first_name'], 'John')
        mock_get_item.assert_called_once_with('users', {'username': {'S': 'johndoe'}}, consistent_read=False, attributes_to_get=None)
        with mock.patch.object(self.users.connection, 'get_item', return_value=expected) as mock_get_item:
            item = self.users.get_item(username='johndoe', attributes=['username', 'first_name'])
        mock_get_item.assert_called_once_with('users', {'username': {'S': 'johndoe'}}, consistent_read=False, attributes_to_get=['username', 'first_name'])

    def test_has_item(self):
        expected = {'Item': {'username': {'S': 'johndoe'}, 'first_name': {'S': 'John'}, 'last_name': {'S': 'Doe'}, 'date_joined': {'N': '1366056668'}, 'friend_count': {'N': '3'}, 'friends': {'SS': ['alice', 'bob', 'jane']}}}
        with mock.patch.object(self.users.connection, 'get_item', return_value=expected) as mock_get_item:
            found = self.users.has_item(username='johndoe')
            self.assertTrue(found)
        with mock.patch.object(self.users.connection, 'get_item') as mock_get_item:
            mock_get_item.side_effect = JSONResponseError('Nope.', None, None)
            found = self.users.has_item(username='mrsmith')
            self.assertFalse(found)

    def test_lookup_hash(self):
        """Tests the "lookup" function with just a hash key"""
        expected = {'Item': {'username': {'S': 'johndoe'}, 'first_name': {'S': 'John'}, 'last_name': {'S': 'Doe'}, 'date_joined': {'N': '1366056668'}, 'friend_count': {'N': '3'}, 'friends': {'SS': ['alice', 'bob', 'jane']}}}
        self.users.schema = [HashKey('username'), RangeKey('date_joined', data_type=NUMBER)]
        with mock.patch.object(self.users, 'get_item', return_value=expected) as mock_get_item:
            self.users.lookup('johndoe')
        mock_get_item.assert_called_once_with(username='johndoe')

    def test_lookup_hash_and_range(self):
        """Test the "lookup" function with a hash and range key"""
        expected = {'Item': {'username': {'S': 'johndoe'}, 'first_name': {'S': 'John'}, 'last_name': {'S': 'Doe'}, 'date_joined': {'N': '1366056668'}, 'friend_count': {'N': '3'}, 'friends': {'SS': ['alice', 'bob', 'jane']}}}
        self.users.schema = [HashKey('username'), RangeKey('date_joined', data_type=NUMBER)]
        with mock.patch.object(self.users, 'get_item', return_value=expected) as mock_get_item:
            self.users.lookup('johndoe', 1366056668)
        mock_get_item.assert_called_once_with(username='johndoe', date_joined=1366056668)

    def test_put_item(self):
        with mock.patch.object(self.users.connection, 'put_item', return_value={}) as mock_put_item:
            self.users.put_item(data={'username': 'johndoe', 'last_name': 'Doe', 'date_joined': 12345})
        mock_put_item.assert_called_once_with('users', {'username': {'S': 'johndoe'}, 'last_name': {'S': 'Doe'}, 'date_joined': {'N': '12345'}}, expected={'username': {'Exists': False}, 'last_name': {'Exists': False}, 'date_joined': {'Exists': False}})

    def test_private_put_item(self):
        with mock.patch.object(self.users.connection, 'put_item', return_value={}) as mock_put_item:
            self.users._put_item({'some': 'data'})
        mock_put_item.assert_called_once_with('users', {'some': 'data'})

    def test_private_update_item(self):
        with mock.patch.object(self.users.connection, 'update_item', return_value={}) as mock_update_item:
            self.users._update_item({'username': 'johndoe'}, {'some': 'data'})
        mock_update_item.assert_called_once_with('users', {'username': {'S': 'johndoe'}}, {'some': 'data'})

    def test_delete_item(self):
        with mock.patch.object(self.users.connection, 'delete_item', return_value={}) as mock_delete_item:
            self.assertTrue(self.users.delete_item(username='johndoe', date_joined=23456))
        mock_delete_item.assert_called_once_with('users', {'username': {'S': 'johndoe'}, 'date_joined': {'N': '23456'}}, expected=None, conditional_operator=None)

    def test_delete_item_conditionally(self):
        with mock.patch.object(self.users.connection, 'delete_item', return_value={}) as mock_delete_item:
            self.assertTrue(self.users.delete_item(expected={'balance__eq': 0}, username='johndoe', date_joined=23456))
        mock_delete_item.assert_called_once_with('users', {'username': {'S': 'johndoe'}, 'date_joined': {'N': '23456'}}, expected={'balance': {'ComparisonOperator': 'EQ', 'AttributeValueList': [{'N': '0'}]}}, conditional_operator=None)

        def side_effect(*args, **kwargs):
            raise exceptions.ConditionalCheckFailedException(400, '', {})
        with mock.patch.object(self.users.connection, 'delete_item', side_effect=side_effect) as mock_delete_item:
            self.assertFalse(self.users.delete_item(expected={'balance__eq': 0}, username='johndoe', date_joined=23456))

    def test_get_key_fields_no_schema_populated(self):
        expected = {'Table': {'AttributeDefinitions': [{'AttributeName': 'username', 'AttributeType': 'S'}, {'AttributeName': 'date_joined', 'AttributeType': 'N'}], 'ItemCount': 5, 'KeySchema': [{'AttributeName': 'username', 'KeyType': 'HASH'}, {'AttributeName': 'date_joined', 'KeyType': 'RANGE'}], 'LocalSecondaryIndexes': [{'IndexName': 'UsernameIndex', 'KeySchema': [{'AttributeName': 'username', 'KeyType': 'HASH'}], 'Projection': {'ProjectionType': 'KEYS_ONLY'}}], 'ProvisionedThroughput': {'ReadCapacityUnits': 20, 'WriteCapacityUnits': 6}, 'TableName': 'Thread', 'TableStatus': 'ACTIVE'}}
        with mock.patch.object(self.users.connection, 'describe_table', return_value=expected) as mock_describe:
            self.assertEqual(self.users.schema, None)
            key_fields = self.users.get_key_fields()
            self.assertEqual(key_fields, ['username', 'date_joined'])
            self.assertEqual(len(self.users.schema), 2)
        mock_describe.assert_called_once_with('users')

    def test_batch_write_no_writes(self):
        with mock.patch.object(self.users.connection, 'batch_write_item', return_value={}) as mock_batch:
            with self.users.batch_write() as batch:
                pass
        self.assertFalse(mock_batch.called)

    def test_batch_write(self):
        with mock.patch.object(self.users.connection, 'batch_write_item', return_value={}) as mock_batch:
            with self.users.batch_write() as batch:
                batch.put_item(data={'username': 'jane', 'date_joined': 12342547})
                batch.delete_item(username='johndoe')
                batch.put_item(data={'username': 'alice', 'date_joined': 12342888})
        mock_batch.assert_called_once_with({'users': [{'PutRequest': {'Item': {'username': {'S': 'jane'}, 'date_joined': {'N': '12342547'}}}}, {'PutRequest': {'Item': {'username': {'S': 'alice'}, 'date_joined': {'N': '12342888'}}}}, {'DeleteRequest': {'Key': {'username': {'S': 'johndoe'}}}}]})

    def test_batch_write_dont_swallow_exceptions(self):
        with mock.patch.object(self.users.connection, 'batch_write_item', return_value={}) as mock_batch:
            try:
                with self.users.batch_write() as batch:
                    raise Exception('OH NOES')
            except Exception as e:
                self.assertEqual(str(e), 'OH NOES')
        self.assertFalse(mock_batch.called)

    def test_batch_write_flushing(self):
        with mock.patch.object(self.users.connection, 'batch_write_item', return_value={}) as mock_batch:
            with self.users.batch_write() as batch:
                batch.put_item(data={'username': 'jane', 'date_joined': 12342547})
                batch.delete_item(username='johndoe1')
                batch.delete_item(username='johndoe2')
                batch.delete_item(username='johndoe3')
                batch.delete_item(username='johndoe4')
                batch.delete_item(username='johndoe5')
                batch.delete_item(username='johndoe6')
                batch.delete_item(username='johndoe7')
                batch.delete_item(username='johndoe8')
                batch.delete_item(username='johndoe9')
                batch.delete_item(username='johndoe10')
                batch.delete_item(username='johndoe11')
                batch.delete_item(username='johndoe12')
                batch.delete_item(username='johndoe13')
                batch.delete_item(username='johndoe14')
                batch.delete_item(username='johndoe15')
                batch.delete_item(username='johndoe16')
                batch.delete_item(username='johndoe17')
                batch.delete_item(username='johndoe18')
                batch.delete_item(username='johndoe19')
                batch.delete_item(username='johndoe20')
                batch.delete_item(username='johndoe21')
                batch.delete_item(username='johndoe22')
                batch.delete_item(username='johndoe23')
                self.assertEqual(mock_batch.call_count, 0)
                batch.delete_item(username='johndoe24')
                self.assertEqual(mock_batch.call_count, 1)
                batch.delete_item(username='johndoe25')
        self.assertEqual(mock_batch.call_count, 2)

    def test_batch_write_unprocessed_items(self):
        unprocessed = {'UnprocessedItems': {'users': [{'PutRequest': {'username': {'S': 'jane'}, 'date_joined': {'N': 12342547}}}]}}
        with mock.patch.object(self.users.connection, 'batch_write_item', return_value=unprocessed) as mock_batch:
            with self.users.batch_write() as batch:
                self.assertEqual(len(batch._unprocessed), 0)
                batch.resend_unprocessed = lambda: True
                batch.put_item(data={'username': 'jane', 'date_joined': 12342547})
                batch.delete_item(username='johndoe')
                batch.put_item(data={'username': 'alice', 'date_joined': 12342888})
            self.assertEqual(len(batch._unprocessed), 1)
        with mock.patch.object(self.users.connection, 'batch_write_item', return_value={}) as mock_batch:
            with self.users.batch_write() as batch:
                self.assertEqual(len(batch._unprocessed), 0)
                batch._unprocessed = [{'PutRequest': {'username': {'S': 'jane'}, 'date_joined': {'N': 12342547}}}]
                batch.put_item(data={'username': 'jane', 'date_joined': 12342547})
                batch.delete_item(username='johndoe')
                batch.put_item(data={'username': 'alice', 'date_joined': 12342888})
                batch.flush()
                self.assertEqual(len(batch._unprocessed), 1)
            self.assertEqual(len(batch._unprocessed), 0)

    def test__build_filters(self):
        filters = self.users._build_filters({'username__eq': 'johndoe', 'date_joined__gte': 1234567, 'age__in': [30, 31, 32, 33], 'last_name__between': ['danzig', 'only'], 'first_name__null': False, 'gender__null': True}, using=FILTER_OPERATORS)
        self.assertEqual(filters, {'username': {'AttributeValueList': [{'S': 'johndoe'}], 'ComparisonOperator': 'EQ'}, 'date_joined': {'AttributeValueList': [{'N': '1234567'}], 'ComparisonOperator': 'GE'}, 'age': {'AttributeValueList': [{'N': '30'}, {'N': '31'}, {'N': '32'}, {'N': '33'}], 'ComparisonOperator': 'IN'}, 'last_name': {'AttributeValueList': [{'S': 'danzig'}, {'S': 'only'}], 'ComparisonOperator': 'BETWEEN'}, 'first_name': {'ComparisonOperator': 'NOT_NULL'}, 'gender': {'ComparisonOperator': 'NULL'}})
        self.assertRaises(exceptions.UnknownFilterTypeError, self.users._build_filters, {'darling__die': True})
        q_filters = self.users._build_filters({'username__eq': 'johndoe', 'date_joined__gte': 1234567, 'last_name__between': ['danzig', 'only'], 'gender__beginswith': 'm'}, using=QUERY_OPERATORS)
        self.assertEqual(q_filters, {'username': {'AttributeValueList': [{'S': 'johndoe'}], 'ComparisonOperator': 'EQ'}, 'date_joined': {'AttributeValueList': [{'N': '1234567'}], 'ComparisonOperator': 'GE'}, 'last_name': {'AttributeValueList': [{'S': 'danzig'}, {'S': 'only'}], 'ComparisonOperator': 'BETWEEN'}, 'gender': {'AttributeValueList': [{'S': 'm'}], 'ComparisonOperator': 'BEGINS_WITH'}})
        self.assertRaises(exceptions.UnknownFilterTypeError, self.users._build_filters, {'darling__die': True}, using=QUERY_OPERATORS)
        self.assertRaises(exceptions.UnknownFilterTypeError, self.users._build_filters, {'first_name__null': True}, using=QUERY_OPERATORS)

    def test_private_query(self):
        expected = {'ConsumedCapacity': {'CapacityUnits': 0.5, 'TableName': 'users'}, 'Count': 4, 'Items': [{'username': {'S': 'johndoe'}, 'first_name': {'S': 'John'}, 'last_name': {'S': 'Doe'}, 'date_joined': {'N': '1366056668'}, 'friend_count': {'N': '3'}, 'friends': {'SS': ['alice', 'bob', 'jane']}}, {'username': {'S': 'jane'}, 'first_name': {'S': 'Jane'}, 'last_name': {'S': 'Doe'}, 'date_joined': {'N': '1366057777'}, 'friend_count': {'N': '2'}, 'friends': {'SS': ['alice', 'johndoe']}}, {'username': {'S': 'alice'}, 'first_name': {'S': 'Alice'}, 'last_name': {'S': 'Expert'}, 'date_joined': {'N': '1366056680'}, 'friend_count': {'N': '1'}, 'friends': {'SS': ['jane']}}, {'username': {'S': 'bob'}, 'first_name': {'S': 'Bob'}, 'last_name': {'S': 'Smith'}, 'date_joined': {'N': '1366056888'}, 'friend_count': {'N': '1'}, 'friends': {'SS': ['johndoe']}}], 'ScannedCount': 4}
        with mock.patch.object(self.users.connection, 'query', return_value=expected) as mock_query:
            results = self.users._query(limit=4, reverse=True, username__between=['aaa', 'mmm'])
            usernames = [res['username'] for res in results['results']]
            self.assertEqual(usernames, ['johndoe', 'jane', 'alice', 'bob'])
            self.assertEqual(len(results['results']), 4)
            self.assertEqual(results['last_key'], None)
        mock_query.assert_called_once_with('users', consistent_read=False, scan_index_forward=False, index_name=None, attributes_to_get=None, limit=4, key_conditions={'username': {'AttributeValueList': [{'S': 'aaa'}, {'S': 'mmm'}], 'ComparisonOperator': 'BETWEEN'}}, select=None, query_filter=None, conditional_operator=None)
        expected['LastEvaluatedKey'] = {'username': {'S': 'johndoe'}}
        with mock.patch.object(self.users.connection, 'query', return_value=expected) as mock_query_2:
            results = self.users._query(limit=4, reverse=True, username__between=['aaa', 'mmm'], exclusive_start_key={'username': 'adam'}, consistent=True, query_filter=None, conditional_operator='AND')
            usernames = [res['username'] for res in results['results']]
            self.assertEqual(usernames, ['johndoe', 'jane', 'alice', 'bob'])
            self.assertEqual(len(results['results']), 4)
            self.assertEqual(results['last_key'], {'username': 'johndoe'})
        mock_query_2.assert_called_once_with('users', key_conditions={'username': {'AttributeValueList': [{'S': 'aaa'}, {'S': 'mmm'}], 'ComparisonOperator': 'BETWEEN'}}, index_name=None, attributes_to_get=None, scan_index_forward=False, limit=4, exclusive_start_key={'username': {'S': 'adam'}}, consistent_read=True, select=None, query_filter=None, conditional_operator='AND')

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

    def test_query(self):
        items_1 = {'results': [Item(self.users, data={'username': 'johndoe', 'first_name': 'John', 'last_name': 'Doe'}), Item(self.users, data={'username': 'jane', 'first_name': 'Jane', 'last_name': 'Doe'})], 'last_key': 'jane'}
        results = self.users.query_2(last_name__eq='Doe')
        self.assertTrue(isinstance(results, ResultSet))
        self.assertEqual(len(results._results), 0)
        self.assertEqual(results.the_callable, self.users._query)
        with mock.patch.object(results, 'the_callable', return_value=items_1) as mock_query:
            res_1 = next(results)
            self.assertEqual(len(results._results), 2)
            self.assertEqual(res_1['username'], 'johndoe')
            res_2 = next(results)
            self.assertEqual(res_2['username'], 'jane')
        self.assertEqual(mock_query.call_count, 1)
        items_2 = {'results': [Item(self.users, data={'username': 'foodoe', 'first_name': 'Foo', 'last_name': 'Doe'})]}
        with mock.patch.object(results, 'the_callable', return_value=items_2) as mock_query_2:
            res_3 = next(results)
            self.assertEqual(len(results._results), 1)
            self.assertEqual(res_3['username'], 'foodoe')
            self.assertRaises(StopIteration, results.next)
        self.assertEqual(mock_query_2.call_count, 1)

    def test_query_with_specific_attributes(self):
        items_1 = {'results': [Item(self.users, data={'username': 'johndoe'}), Item(self.users, data={'username': 'jane'})], 'last_key': 'jane'}
        results = self.users.query_2(last_name__eq='Doe', attributes=['username'])
        self.assertTrue(isinstance(results, ResultSet))
        self.assertEqual(len(results._results), 0)
        self.assertEqual(results.the_callable, self.users._query)
        with mock.patch.object(results, 'the_callable', return_value=items_1) as mock_query:
            res_1 = next(results)
            self.assertEqual(len(results._results), 2)
            self.assertEqual(res_1['username'], 'johndoe')
            self.assertEqual(list(res_1.keys()), ['username'])
            res_2 = next(results)
            self.assertEqual(res_2['username'], 'jane')
        self.assertEqual(mock_query.call_count, 1)

    def test_scan(self):
        items_1 = {'results': [Item(self.users, data={'username': 'johndoe', 'first_name': 'John', 'last_name': 'Doe'}), Item(self.users, data={'username': 'jane', 'first_name': 'Jane', 'last_name': 'Doe'})], 'last_key': 'jane'}
        results = self.users.scan(last_name__eq='Doe')
        self.assertTrue(isinstance(results, ResultSet))
        self.assertEqual(len(results._results), 0)
        self.assertEqual(results.the_callable, self.users._scan)
        with mock.patch.object(results, 'the_callable', return_value=items_1) as mock_scan:
            res_1 = next(results)
            self.assertEqual(len(results._results), 2)
            self.assertEqual(res_1['username'], 'johndoe')
            res_2 = next(results)
            self.assertEqual(res_2['username'], 'jane')
        self.assertEqual(mock_scan.call_count, 1)
        items_2 = {'results': [Item(self.users, data={'username': 'zoeydoe', 'first_name': 'Zoey', 'last_name': 'Doe'})]}
        with mock.patch.object(results, 'the_callable', return_value=items_2) as mock_scan_2:
            res_3 = next(results)
            self.assertEqual(len(results._results), 1)
            self.assertEqual(res_3['username'], 'zoeydoe')
            self.assertRaises(StopIteration, results.next)
        self.assertEqual(mock_scan_2.call_count, 1)

    def test_scan_with_specific_attributes(self):
        items_1 = {'results': [Item(self.users, data={'username': 'johndoe'}), Item(self.users, data={'username': 'jane'})], 'last_key': 'jane'}
        results = self.users.scan(attributes=['username'])
        self.assertTrue(isinstance(results, ResultSet))
        self.assertEqual(len(results._results), 0)
        self.assertEqual(results.the_callable, self.users._scan)
        with mock.patch.object(results, 'the_callable', return_value=items_1) as mock_query:
            res_1 = next(results)
            self.assertEqual(len(results._results), 2)
            self.assertEqual(res_1['username'], 'johndoe')
            self.assertEqual(list(res_1.keys()), ['username'])
            res_2 = next(results)
            self.assertEqual(res_2['username'], 'jane')
        self.assertEqual(mock_query.call_count, 1)

    def test_count(self):
        expected = {'Table': {'AttributeDefinitions': [{'AttributeName': 'username', 'AttributeType': 'S'}], 'ItemCount': 5, 'KeySchema': [{'AttributeName': 'username', 'KeyType': 'HASH'}], 'LocalSecondaryIndexes': [{'IndexName': 'UsernameIndex', 'KeySchema': [{'AttributeName': 'username', 'KeyType': 'HASH'}], 'Projection': {'ProjectionType': 'KEYS_ONLY'}}], 'ProvisionedThroughput': {'ReadCapacityUnits': 20, 'WriteCapacityUnits': 6}, 'TableName': 'Thread', 'TableStatus': 'ACTIVE'}}
        with mock.patch.object(self.users, 'describe', return_value=expected) as mock_count:
            self.assertEqual(self.users.count(), 5)

    def test_query_count_simple(self):
        expected_0 = {'Count': 0.0}
        expected_1 = {'Count': 10.0}
        with mock.patch.object(self.users.connection, 'query', return_value=expected_0) as mock_query:
            results = self.users.query_count(username__eq='notmyname')
            self.assertTrue(isinstance(results, int))
            self.assertEqual(results, 0)
        self.assertEqual(mock_query.call_count, 1)
        self.assertIn('scan_index_forward', mock_query.call_args[1])
        self.assertEqual(True, mock_query.call_args[1]['scan_index_forward'])
        self.assertIn('limit', mock_query.call_args[1])
        self.assertEqual(None, mock_query.call_args[1]['limit'])
        with mock.patch.object(self.users.connection, 'query', return_value=expected_1) as mock_query:
            results = self.users.query_count(username__gt='somename', consistent=True, scan_index_forward=False, limit=10)
            self.assertTrue(isinstance(results, int))
            self.assertEqual(results, 10)
        self.assertEqual(mock_query.call_count, 1)
        self.assertIn('scan_index_forward', mock_query.call_args[1])
        self.assertEqual(False, mock_query.call_args[1]['scan_index_forward'])
        self.assertIn('limit', mock_query.call_args[1])
        self.assertEqual(10, mock_query.call_args[1]['limit'])

    def test_query_count_paginated(self):

        def return_side_effect(*args, **kwargs):
            if kwargs.get('exclusive_start_key'):
                return {'Count': 10, 'LastEvaluatedKey': None}
            else:
                return {'Count': 20, 'LastEvaluatedKey': {'username': {'S': 'johndoe'}, 'date_joined': {'N': '4118642633'}}}
        with mock.patch.object(self.users.connection, 'query', side_effect=return_side_effect) as mock_query:
            count = self.users.query_count(username__eq='johndoe')
            self.assertTrue(isinstance(count, int))
            self.assertEqual(30, count)
            self.assertEqual(mock_query.call_count, 2)

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

    def test_batch_get(self):
        items_1 = {'results': [Item(self.users, data={'username': 'johndoe', 'first_name': 'John', 'last_name': 'Doe'}), Item(self.users, data={'username': 'jane', 'first_name': 'Jane', 'last_name': 'Doe'})], 'last_key': None, 'unprocessed_keys': ['zoeydoe']}
        results = self.users.batch_get(keys=[{'username': 'johndoe'}, {'username': 'jane'}, {'username': 'zoeydoe'}])
        self.assertTrue(isinstance(results, BatchGetResultSet))
        self.assertEqual(len(results._results), 0)
        self.assertEqual(results.the_callable, self.users._batch_get)
        with mock.patch.object(results, 'the_callable', return_value=items_1) as mock_batch_get:
            res_1 = next(results)
            self.assertEqual(len(results._results), 2)
            self.assertEqual(res_1['username'], 'johndoe')
            res_2 = next(results)
            self.assertEqual(res_2['username'], 'jane')
        self.assertEqual(mock_batch_get.call_count, 1)
        self.assertEqual(results._keys_left, ['zoeydoe'])
        items_2 = {'results': [Item(self.users, data={'username': 'zoeydoe', 'first_name': 'Zoey', 'last_name': 'Doe'})]}
        with mock.patch.object(results, 'the_callable', return_value=items_2) as mock_batch_get_2:
            res_3 = next(results)
            self.assertEqual(len(results._results), 1)
            self.assertEqual(res_3['username'], 'zoeydoe')
            self.assertRaises(StopIteration, results.next)
        self.assertEqual(mock_batch_get_2.call_count, 1)
        self.assertEqual(results._keys_left, [])