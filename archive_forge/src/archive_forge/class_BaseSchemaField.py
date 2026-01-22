from boto.dynamodb2.types import STRING
class BaseSchemaField(object):
    """
    An abstract class for defining schema fields.

    Contains most of the core functionality for the field. Subclasses must
    define an ``attr_type`` to pass to DynamoDB.
    """
    attr_type = None

    def __init__(self, name, data_type=STRING):
        """
        Creates a Python schema field, to represent the data to pass to
        DynamoDB.

        Requires a ``name`` parameter, which should be a string name of the
        field.

        Optionally accepts a ``data_type`` parameter, which should be a
        constant from ``boto.dynamodb2.types``. (Default: ``STRING``)
        """
        self.name = name
        self.data_type = data_type

    def definition(self):
        """
        Returns the attribute definition structure DynamoDB expects.

        Example::

            >>> field.definition()
            {
                'AttributeName': 'username',
                'AttributeType': 'S',
            }

        """
        return {'AttributeName': self.name, 'AttributeType': self.data_type}

    def schema(self):
        """
        Returns the schema structure DynamoDB expects.

        Example::

            >>> field.schema()
            {
                'AttributeName': 'username',
                'KeyType': 'HASH',
            }

        """
        return {'AttributeName': self.name, 'KeyType': self.attr_type}