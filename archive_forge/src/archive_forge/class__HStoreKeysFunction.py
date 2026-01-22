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
class _HStoreKeysFunction(sqlfunc.GenericFunction):
    type = ARRAY(sqltypes.Text)
    name = 'akeys'
    inherit_cache = True