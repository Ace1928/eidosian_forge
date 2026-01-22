from boto.dynamodb.layer1 import Layer1
from boto.dynamodb.table import Table
from boto.dynamodb.schema import Schema
from boto.dynamodb.item import Item
from boto.dynamodb.batch import BatchList, BatchWriteList
from boto.dynamodb.types import get_dynamodb_type, Dynamizer, \
class TableGenerator(object):
    """
    This is an object that wraps up the table_generator function.
    The only real reason to have this is that we want to be able
    to accumulate and return the ConsumedCapacityUnits element that
    is part of each response.

    :ivar last_evaluated_key: A sequence representing the key(s)
        of the item last evaluated, or None if no additional
        results are available.

    :ivar remaining: The remaining quantity of results requested.

    :ivar table: The table to which the call was made.
    """

    def __init__(self, table, callable, remaining, item_class, kwargs):
        self.table = table
        self.callable = callable
        self.remaining = -1 if remaining is None else remaining
        self.item_class = item_class
        self.kwargs = kwargs
        self._consumed_units = 0.0
        self.last_evaluated_key = None
        self._count = 0
        self._scanned_count = 0
        self._response = None

    @property
    def count(self):
        """
        The total number of items retrieved thus far.  This value changes with
        iteration and even when issuing a call with count=True, it is necessary
        to complete the iteration to assert an accurate count value.
        """
        self.response
        return self._count

    @property
    def scanned_count(self):
        """
        As above, but representing the total number of items scanned by
        DynamoDB, without regard to any filters.
        """
        self.response
        return self._scanned_count

    @property
    def consumed_units(self):
        """
        Returns a float representing the ConsumedCapacityUnits accumulated.
        """
        self.response
        return self._consumed_units

    @property
    def response(self):
        """
        The current response to the call from DynamoDB.
        """
        return self.next_response() if self._response is None else self._response

    def next_response(self):
        """
        Issue a call and return the result.  You can invoke this method
        while iterating over the TableGenerator in order to skip to the
        next "page" of results.
        """
        limit = self.kwargs.get('limit')
        if self.remaining > 0 and (limit is None or limit > self.remaining):
            self.kwargs['limit'] = self.remaining
        self._response = self.callable(**self.kwargs)
        self.kwargs['limit'] = limit
        self._consumed_units += self._response.get('ConsumedCapacityUnits', 0.0)
        self._count += self._response.get('Count', 0)
        self._scanned_count += self._response.get('ScannedCount', 0)
        if 'LastEvaluatedKey' in self._response:
            lek = self._response['LastEvaluatedKey']
            esk = self.table.layer2.dynamize_last_evaluated_key(lek)
            self.kwargs['exclusive_start_key'] = esk
            lektuple = (lek['HashKeyElement'],)
            if 'RangeKeyElement' in lek:
                lektuple += (lek['RangeKeyElement'],)
            self.last_evaluated_key = lektuple
        else:
            self.last_evaluated_key = None
        return self._response

    def __iter__(self):
        while self.remaining != 0:
            response = self.response
            for item in response.get('Items', []):
                self.remaining -= 1
                yield self.item_class(self.table, attrs=item)
                if self.remaining == 0:
                    break
                if response is not self._response:
                    break
            else:
                if self.last_evaluated_key is not None:
                    self.next_response()
                    continue
                break
            if response is not self._response:
                continue
            break