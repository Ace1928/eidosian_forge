from .array import ARRAY
from .array import array as _pg_array
from .operators import ASTEXT
from .operators import CONTAINED_BY
from .operators import CONTAINS
from .operators import DELETE_PATH
from .operators import HAS_ALL
from .operators import HAS_ANY
from .operators import HAS_KEY
from .operators import JSONPATH_ASTEXT
from .operators import PATH_EXISTS
from .operators import PATH_MATCH
from ... import types as sqltypes
from ...sql import cast
def _processor(self, dialect, super_proc):

    def process(value):
        if isinstance(value, str):
            return value
        elif value:
            value = '{%s}' % ', '.join(map(str, value))
        else:
            value = '{}'
        if super_proc:
            value = super_proc(value)
        return value
    return process