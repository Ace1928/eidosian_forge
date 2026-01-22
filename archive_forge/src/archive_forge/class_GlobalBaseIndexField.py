from boto.dynamodb2.types import STRING
class GlobalBaseIndexField(BaseIndexField):
    """
    An abstract class for defining global indexes.

    Contains most of the core functionality for the index. Subclasses must
    define a ``projection_type`` to pass to DynamoDB.
    """
    throughput = {'read': 5, 'write': 5}

    def __init__(self, *args, **kwargs):
        throughput = kwargs.pop('throughput', None)
        if throughput is not None:
            self.throughput = throughput
        super(GlobalBaseIndexField, self).__init__(*args, **kwargs)

    def schema(self):
        """
        Returns the schema structure DynamoDB expects.

        Example::

            >>> index.schema()
            {
                'IndexName': 'LastNameIndex',
                'KeySchema': [
                    {
                        'AttributeName': 'username',
                        'KeyType': 'HASH',
                    },
                ],
                'Projection': {
                    'ProjectionType': 'KEYS_ONLY',
                },
                'ProvisionedThroughput': {
                    'ReadCapacityUnits': 5,
                    'WriteCapacityUnits': 5
                }
            }

        """
        schema_data = super(GlobalBaseIndexField, self).schema()
        schema_data['ProvisionedThroughput'] = {'ReadCapacityUnits': int(self.throughput['read']), 'WriteCapacityUnits': int(self.throughput['write'])}
        return schema_data