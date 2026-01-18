import os
import time
from tests.unit import unittest
from boto.dynamodb2 import exceptions
from boto.dynamodb2.fields import (HashKey, RangeKey, KeysOnlyIndex,
from boto.dynamodb2.items import Item
from boto.dynamodb2.table import Table
from boto.dynamodb2.types import NUMBER, STRING
def test_unprocessed_batch_writes(self):
    users = Table.create('slow_users', schema=[HashKey('user_id')], throughput={'read': 1, 'write': 1})
    self.addCleanup(users.delete)
    time.sleep(60)
    with users.batch_write() as batch:
        for i in range(500):
            batch.put_item(data={'user_id': str(i), 'name': 'Droid #{0}'.format(i)})
        self.assertTrue(len(batch._unprocessed) > 0)
    self.assertEqual(len(batch._unprocessed), 0)