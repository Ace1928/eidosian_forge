import boto
from boto.dynamodb2 import exceptions
from boto.dynamodb2.fields import (HashKey, RangeKey,
from boto.dynamodb2.items import Item
from boto.dynamodb2.layer1 import DynamoDBConnection
from boto.dynamodb2.results import ResultSet, BatchGetResultSet
from boto.dynamodb2.types import (NonBooleanDynamizer, Dynamizer, FILTER_OPERATORS,
from boto.exception import JSONResponseError
def _introspect_global_indexes(self, raw_global_indexes):
    """
        Given a raw global index structure back from a DynamoDB response, parse
        out & build the high-level Python objects that represent them.
        """
    return self._introspect_all_indexes(raw_global_indexes, self._PROJECTION_TYPE_TO_INDEX.get('global_indexes'))