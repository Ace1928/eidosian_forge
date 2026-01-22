from boto.dynamodb2.types import STRING
class HashKey(BaseSchemaField):
    """
    An field representing a hash key.

    Example::

        >>> from boto.dynamodb2.types import NUMBER
        >>> HashKey('username')
        >>> HashKey('date_joined', data_type=NUMBER)

    """
    attr_type = 'HASH'