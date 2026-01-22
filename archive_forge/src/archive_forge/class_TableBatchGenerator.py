from boto.dynamodb.batch import BatchList
from boto.dynamodb.schema import Schema
from boto.dynamodb.item import Item
from boto.dynamodb import exceptions as dynamodb_exceptions
import time
class TableBatchGenerator(object):
    """
    A low-level generator used to page through results from
    batch_get_item operations.

    :ivar consumed_units: An integer that holds the number of
        ConsumedCapacityUnits accumulated thus far for this
        generator.
    """

    def __init__(self, table, keys, attributes_to_get=None, consistent_read=False):
        self.table = table
        self.keys = keys
        self.consumed_units = 0
        self.attributes_to_get = attributes_to_get
        self.consistent_read = consistent_read

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

    def __iter__(self):
        while self.keys:
            batch = BatchList(self.table.layer2)
            batch.add_batch(self.table, self.keys[:100], self.attributes_to_get)
            res = batch.submit()
            if self.table.name not in res[u'Responses']:
                continue
            self.consumed_units += res[u'Responses'][self.table.name][u'ConsumedCapacityUnits']
            for elem in res[u'Responses'][self.table.name][u'Items']:
                yield elem
            self.keys = self.keys[100:]
            self._queue_unprocessed(res)