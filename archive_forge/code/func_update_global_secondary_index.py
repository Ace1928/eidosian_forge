import boto
from boto.dynamodb2 import exceptions
from boto.dynamodb2.fields import (HashKey, RangeKey,
from boto.dynamodb2.items import Item
from boto.dynamodb2.layer1 import DynamoDBConnection
from boto.dynamodb2.results import ResultSet, BatchGetResultSet
from boto.dynamodb2.types import (NonBooleanDynamizer, Dynamizer, FILTER_OPERATORS,
from boto.exception import JSONResponseError
def update_global_secondary_index(self, global_indexes):
    """
        Updates a global index(es) in DynamoDB after the table has been created.

        Requires a ``global_indexes`` parameter, which should be a
        dictionary. If provided, it should specify the index name, which is also
        a dict containing a ``read`` & ``write`` key, both of which
        should have an integer value associated with them.

        To update ``global_indexes`` information on the ``Table``, you'll need
        to call ``Table.describe``.

        Returns ``True`` on success.

        Example::

            # To update a global index
            >>> users.update_global_secondary_index(global_indexes={
            ...     'TheIndexNameHere': {
            ...         'read': 15,
            ...         'write': 5,
            ...     }
            ... })
            True

        """
    if global_indexes:
        gsi_data = []
        for gsi_name, gsi_throughput in global_indexes.items():
            gsi_data.append({'Update': {'IndexName': gsi_name, 'ProvisionedThroughput': {'ReadCapacityUnits': int(gsi_throughput['read']), 'WriteCapacityUnits': int(gsi_throughput['write'])}}})
        self.connection.update_table(self.table_name, global_secondary_index_updates=gsi_data)
        return True
    else:
        msg = 'You need to provide the global indexes to update_global_secondary_index method'
        boto.log.error(msg)
        return False