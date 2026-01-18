import os
import time
from tests.unit import unittest
from boto.dynamodb2 import exceptions
from boto.dynamodb2.fields import (HashKey, RangeKey, KeysOnlyIndex,
from boto.dynamodb2.items import Item
from boto.dynamodb2.table import Table
from boto.dynamodb2.types import NUMBER, STRING
def test_update_table_online_indexing_support(self):
    users = Table.create('online_indexing_support_users', schema=[HashKey('user_id')], throughput={'read': 5, 'write': 5}, global_indexes=[GlobalAllIndex('EmailGSIIndex', parts=[HashKey('email')], throughput={'read': 2, 'write': 2})])
    self.addCleanup(users.delete)
    time.sleep(60)
    users.describe()
    self.assertEqual(len(users.global_indexes), 1)
    self.assertEqual(users.global_indexes[0].throughput['read'], 2)
    self.assertEqual(users.global_indexes[0].throughput['write'], 2)
    users.update_global_secondary_index(global_indexes={'EmailGSIIndex': {'read': 2, 'write': 1}})
    time.sleep(60)
    users.describe()
    self.assertEqual(len(users.global_indexes), 1)
    self.assertEqual(users.global_indexes[0].throughput['read'], 2)
    self.assertEqual(users.global_indexes[0].throughput['write'], 1)
    users.update(global_indexes={'EmailGSIIndex': {'read': 3, 'write': 2}})
    time.sleep(60)
    users.describe()
    self.assertEqual(len(users.global_indexes), 1)
    self.assertEqual(users.global_indexes[0].throughput['read'], 3)
    self.assertEqual(users.global_indexes[0].throughput['write'], 2)
    users.delete_global_secondary_index('EmailGSIIndex')
    time.sleep(60)
    users.describe()
    self.assertEqual(len(users.global_indexes), 0)
    users.create_global_secondary_index(global_index=GlobalAllIndex('AddressGSIIndex', parts=[HashKey('address', data_type=STRING)], throughput={'read': 1, 'write': 1}))
    time.sleep(60 * 10)
    users.describe()
    self.assertEqual(len(users.global_indexes), 1)
    self.assertEqual(users.global_indexes[0].throughput['read'], 1)
    self.assertEqual(users.global_indexes[0].throughput['write'], 1)