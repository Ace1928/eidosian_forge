import os
import time
from tests.unit import unittest
from boto.dynamodb2 import exceptions
from boto.dynamodb2.fields import (HashKey, RangeKey, KeysOnlyIndex,
from boto.dynamodb2.items import Item
from boto.dynamodb2.table import Table
from boto.dynamodb2.types import NUMBER, STRING
def test_query_with_limits(self):
    posts = Table.create('posts', schema=[HashKey('thread'), RangeKey('posted_on')], throughput={'read': 5, 'write': 5})
    self.addCleanup(posts.delete)
    time.sleep(60)
    test_data_path = os.path.join(os.path.dirname(__file__), 'forum_test_data.json')
    with open(test_data_path, 'r') as test_data:
        data = json.load(test_data)
        with posts.batch_write() as batch:
            for post in data:
                batch.put_item(post)
    time.sleep(5)
    results = posts.query_2(thread__eq='Favorite chiptune band?', posted_on__gte='2013-12-24T00:00:00', max_page_size=2)
    all_posts = list(results)
    self.assertEqual([post['posted_by'] for post in all_posts], ['joe', 'jane', 'joe', 'joe', 'jane', 'joe'])
    self.assertTrue(results._fetches >= 3)