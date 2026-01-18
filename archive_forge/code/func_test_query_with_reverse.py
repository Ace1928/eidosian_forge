import os
import time
from tests.unit import unittest
from boto.dynamodb2 import exceptions
from boto.dynamodb2.fields import (HashKey, RangeKey, KeysOnlyIndex,
from boto.dynamodb2.items import Item
from boto.dynamodb2.table import Table
from boto.dynamodb2.types import NUMBER, STRING
def test_query_with_reverse(self):
    posts = Table.create('more-posts', schema=[HashKey('thread'), RangeKey('posted_on')], throughput={'read': 5, 'write': 5})
    self.addCleanup(posts.delete)
    time.sleep(60)
    test_data_path = os.path.join(os.path.dirname(__file__), 'forum_test_data.json')
    with open(test_data_path, 'r') as test_data:
        data = json.load(test_data)
        with posts.batch_write() as batch:
            for post in data:
                batch.put_item(post)
    time.sleep(5)
    results = posts.query_2(thread__eq='Favorite chiptune band?', posted_on__gte='2013-12-24T00:00:00')
    self.assertEqual([post['posted_on'] for post in results], ['2013-12-24T12:30:54', '2013-12-24T12:35:40', '2013-12-24T13:45:30', '2013-12-24T14:15:14', '2013-12-24T14:25:33', '2013-12-24T15:22:22'])
    results = posts.query_2(thread__eq='Favorite chiptune band?', posted_on__gte='2013-12-24T00:00:00', reverse=False)
    self.assertEqual([post['posted_on'] for post in results], ['2013-12-24T12:30:54', '2013-12-24T12:35:40', '2013-12-24T13:45:30', '2013-12-24T14:15:14', '2013-12-24T14:25:33', '2013-12-24T15:22:22'])
    results = posts.query_2(thread__eq='Favorite chiptune band?', posted_on__gte='2013-12-24T00:00:00', reverse=True)
    self.assertEqual([post['posted_on'] for post in results], ['2013-12-24T15:22:22', '2013-12-24T14:25:33', '2013-12-24T14:15:14', '2013-12-24T13:45:30', '2013-12-24T12:35:40', '2013-12-24T12:30:54'])
    results = posts.query(thread__eq='Favorite chiptune band?', posted_on__gte='2013-12-24T00:00:00')
    self.assertEqual([post['posted_on'] for post in results], ['2013-12-24T15:22:22', '2013-12-24T14:25:33', '2013-12-24T14:15:14', '2013-12-24T13:45:30', '2013-12-24T12:35:40', '2013-12-24T12:30:54'])
    results = posts.query(thread__eq='Favorite chiptune band?', posted_on__gte='2013-12-24T00:00:00', reverse=True)
    self.assertEqual([post['posted_on'] for post in results], ['2013-12-24T12:30:54', '2013-12-24T12:35:40', '2013-12-24T13:45:30', '2013-12-24T14:15:14', '2013-12-24T14:25:33', '2013-12-24T15:22:22'])