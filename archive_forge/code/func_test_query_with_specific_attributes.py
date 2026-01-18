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