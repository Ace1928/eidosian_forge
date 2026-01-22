import boto
from boto.dynamodb2 import exceptions
from boto.dynamodb2.fields import (HashKey, RangeKey,
from boto.dynamodb2.items import Item
from boto.dynamodb2.layer1 import DynamoDBConnection
from boto.dynamodb2.results import ResultSet, BatchGetResultSet
from boto.dynamodb2.types import (NonBooleanDynamizer, Dynamizer, FILTER_OPERATORS,
from boto.exception import JSONResponseError
def _encode_keys(self, keys):
    """
        Given a flat Python dictionary of keys/values, converts it into the
        nested dictionary DynamoDB expects.

        Converts::

            {
                'username': 'john',
                'tags': [1, 2, 5],
            }

        ...to...::

            {
                'username': {'S': 'john'},
                'tags': {'NS': ['1', '2', '5']},
            }

        """
    raw_key = {}
    for key, value in keys.items():
        raw_key[key] = self._dynamizer.encode(value)
    return raw_key