import re
from .types import _StringType
from ... import exc
from ... import sql
from ... import util
from ...sql import sqltypes
@classmethod
def adapt_emulated_to_native(cls, impl, **kw):
    """Produce a MySQL native :class:`.mysql.ENUM` from plain
        :class:`.Enum`.

        """
    kw.setdefault('validate_strings', impl.validate_strings)
    kw.setdefault('values_callable', impl.values_callable)
    kw.setdefault('omit_aliases', impl._omit_aliases)
    return cls(**kw)