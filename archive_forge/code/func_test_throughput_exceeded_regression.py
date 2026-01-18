import time
from tests.unit import unittest
from boto.dynamodb2 import exceptions
from boto.dynamodb2.layer1 import DynamoDBConnection
def test_throughput_exceeded_regression(self):
    tiny_tablename = 'TinyThroughput'
    tiny = self.create_table(tiny_tablename, self.attributes, self.schema, {'ReadCapacityUnits': 1, 'WriteCapacityUnits': 1})
    self.dynamodb.put_item(tiny_tablename, {'username': {'S': 'johndoe'}, 'first_name': {'S': 'John'}, 'last_name': {'S': 'Doe'}, 'date_joined': {'N': '1366056668'}})
    self.dynamodb.put_item(tiny_tablename, {'username': {'S': 'jane'}, 'first_name': {'S': 'Jane'}, 'last_name': {'S': 'Doe'}, 'date_joined': {'N': '1366056669'}})
    self.dynamodb.put_item(tiny_tablename, {'username': {'S': 'alice'}, 'first_name': {'S': 'Alice'}, 'last_name': {'S': 'Expert'}, 'date_joined': {'N': '1366057000'}})
    time.sleep(20)
    for i in range(100):
        self.dynamodb.scan(tiny_tablename)