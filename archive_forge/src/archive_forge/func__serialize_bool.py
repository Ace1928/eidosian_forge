from .. import errors
from ..osutils import basename
from ..revision import NULL_REVISION
from . import inventory
def _serialize_bool(self, value):
    if value:
        return b'true'
    else:
        return b'false'