import json
from sqlalchemy.dialects import mysql
from sqlalchemy.types import Integer, Text, TypeDecorator
class JsonEncodedType(TypeDecorator):
    """Base column type for data serialized as JSON-encoded string in db."""
    type = None
    impl = Text
    cache_ok = True
    'This type is safe to cache.'

    def __init__(self, mysql_as_long=False, mysql_as_medium=False):
        """Initialize JSON-encoding type."""
        super(JsonEncodedType, self).__init__()
        if mysql_as_long and mysql_as_medium:
            raise TypeError('mysql_as_long and mysql_as_medium are mutually exclusive')
        if mysql_as_long:
            self.impl = Text().with_variant(mysql.LONGTEXT(), 'mysql')
        elif mysql_as_medium:
            self.impl = Text().with_variant(mysql.MEDIUMTEXT(), 'mysql')

    def process_bind_param(self, value, dialect):
        """Bind parameters to the process."""
        if value is None:
            if self.type is not None:
                value = self.type()
        elif self.type is not None and (not isinstance(value, self.type)):
            raise TypeError('%s supposes to store %s objects, but %s given' % (self.__class__.__name__, self.type.__name__, type(value).__name__))
        serialized_value = json.dumps(value)
        return serialized_value

    def process_result_value(self, value, dialect):
        """Process result value."""
        if value is not None:
            value = json.loads(value)
        return value