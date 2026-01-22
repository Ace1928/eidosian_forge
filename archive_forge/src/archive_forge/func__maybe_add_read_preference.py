from __future__ import annotations
import datetime
import random
import struct
from io import BytesIO as _BytesIO
from typing import (
import bson
from bson import CodecOptions, _decode_selective, _dict_to_bson, _make_c_string, encode
from bson.int64 import Int64
from bson.raw_bson import (
from bson.son import SON
from pymongo.errors import (
from pymongo.hello import HelloCompat
from pymongo.helpers import _handle_reauth
from pymongo.read_preferences import ReadPreference
from pymongo.write_concern import WriteConcern
def _maybe_add_read_preference(spec: MutableMapping[str, Any], read_preference: _ServerMode) -> MutableMapping[str, Any]:
    """Add $readPreference to spec when appropriate."""
    mode = read_preference.mode
    document = read_preference.document
    if mode and (mode != ReadPreference.SECONDARY_PREFERRED.mode or len(document) > 1):
        if '$query' not in spec:
            spec = SON([('$query', spec)])
        spec['$readPreference'] = document
    return spec