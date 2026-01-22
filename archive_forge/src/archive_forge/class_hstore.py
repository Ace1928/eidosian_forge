import re
from .array import ARRAY
from .operators import CONTAINED_BY
from .operators import CONTAINS
from .operators import GETITEM
from .operators import HAS_ALL
from .operators import HAS_ANY
from .operators import HAS_KEY
from ... import types as sqltypes
from ...sql import functions as sqlfunc
class hstore(sqlfunc.GenericFunction):
    """Construct an hstore value within a SQL expression using the
    PostgreSQL ``hstore()`` function.

    The :class:`.hstore` function accepts one or two arguments as described
    in the PostgreSQL documentation.

    E.g.::

        from sqlalchemy.dialects.postgresql import array, hstore

        select(hstore('key1', 'value1'))

        select(
            hstore(
                array(['key1', 'key2', 'key3']),
                array(['value1', 'value2', 'value3'])
            )
        )

    .. seealso::

        :class:`.HSTORE` - the PostgreSQL ``HSTORE`` datatype.

    """
    type = HSTORE
    name = 'hstore'
    inherit_cache = True