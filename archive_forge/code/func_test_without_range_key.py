import time
from tests.unit import unittest
from boto.dynamodb2 import exceptions
from boto.dynamodb2.layer1 import DynamoDBConnection
def test_without_range_key(self):
    result = self.create_table(self.table_name, [{'AttributeName': self.hash_key_name, 'AttributeType': self.hash_key_type}], [{'AttributeName': self.hash_key_name, 'KeyType': 'HASH'}], self.provisioned_throughput)
    self.assertEqual(result['TableDescription']['TableName'], self.table_name)
    description = self.dynamodb.describe_table(self.table_name)
    self.assertEqual(description['Table']['ItemCount'], 0)
    record_1_data = {'username': {'S': 'johndoe'}, 'first_name': {'S': 'John'}, 'last_name': {'S': 'Doe'}, 'date_joined': {'N': '1366056668'}, 'friend_count': {'N': '3'}, 'friends': {'SS': ['alice', 'bob', 'jane']}}
    r1_result = self.dynamodb.put_item(self.table_name, record_1_data)
    johndoe = self.dynamodb.get_item(self.table_name, key={'username': {'S': 'johndoe'}}, consistent_read=True)
    self.assertEqual(johndoe['Item']['username']['S'], 'johndoe')
    self.assertEqual(johndoe['Item']['first_name']['S'], 'John')
    self.assertEqual(johndoe['Item']['friends']['SS'], ['alice', 'bob', 'jane'])