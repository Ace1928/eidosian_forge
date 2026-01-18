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
def test_partial_no_changes(self):
    with mock.patch.object(self.table, '_update_item', return_value=True) as mock_update_item:
        self.johndoe.mark_clean()
        self.assertFalse(self.johndoe.partial_save())
    self.assertFalse(mock_update_item.called)