import time
from tests.unit import unittest
from boto.dynamodb.layer2 import Layer2
from boto.dynamodb.table import Table
from boto.dynamodb.schema import Schema
class TestDynamoDBTable(unittest.TestCase):
    dynamodb = True

    def setUp(self):
        self.dynamodb = Layer2()
        self.schema = Schema.create(('foo', 'N'), ('bar', 'S'))
        self.table_name = 'testtable%s' % int(time.time())

    def create_table(self, table_name, schema, read_units, write_units):
        result = self.dynamodb.create_table(table_name, schema, read_units, write_units)
        self.addCleanup(self.dynamodb.delete_table, result)
        return result

    def assertAllEqual(self, *items):
        first = items[0]
        for item in items[1:]:
            self.assertEqual(first, item)

    def test_table_retrieval_parity(self):
        created_table = self.dynamodb.create_table(self.table_name, self.schema, 1, 1)
        created_table.refresh(wait_for_active=True)
        retrieved_table = self.dynamodb.get_table(self.table_name)
        constructed_table = self.dynamodb.table_from_schema(self.table_name, self.schema)
        self.assertAllEqual(created_table.name, retrieved_table.name, constructed_table.name)
        self.assertAllEqual(created_table.schema, retrieved_table.schema, constructed_table.schema)
        self.assertEqual(created_table.create_time, retrieved_table.create_time)
        self.assertEqual(created_table.status, retrieved_table.status)
        self.assertEqual(created_table.read_units, retrieved_table.read_units)
        self.assertEqual(created_table.write_units, retrieved_table.write_units)
        self.assertIsNone(constructed_table.create_time)
        self.assertIsNone(constructed_table.status)
        self.assertIsNone(constructed_table.read_units)
        self.assertIsNone(constructed_table.write_units)