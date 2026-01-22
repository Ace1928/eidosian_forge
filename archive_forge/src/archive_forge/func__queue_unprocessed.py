from boto.dynamodb.batch import BatchList
from boto.dynamodb.schema import Schema
from boto.dynamodb.item import Item
from boto.dynamodb import exceptions as dynamodb_exceptions
import time
def _queue_unprocessed(self, res):
    if u'UnprocessedKeys' not in res:
        return
    if self.table.name not in res[u'UnprocessedKeys']:
        return
    keys = res[u'UnprocessedKeys'][self.table.name][u'Keys']
    for key in keys:
        h = key[u'HashKeyElement']
        r = key[u'RangeKeyElement'] if u'RangeKeyElement' in key else None
        self.keys.append((h, r))