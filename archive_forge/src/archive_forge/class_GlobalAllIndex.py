from boto.dynamodb2.types import STRING
class GlobalAllIndex(GlobalBaseIndexField):
    """
    An index signifying all fields should be in the index.

    Example::

        >>> GlobalAllIndex('MostRecentlyJoined', parts=[
        ...     HashKey('username'),
        ...     RangeKey('date_joined')
        ... ],
        ... throughput={
        ...     'read': 2,
        ...     'write': 1,
        ... })

    """
    projection_type = 'ALL'