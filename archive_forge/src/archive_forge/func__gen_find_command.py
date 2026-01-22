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
def _gen_find_command(coll: str, spec: Mapping[str, Any], projection: Optional[Union[Mapping[str, Any], Iterable[str]]], skip: int, limit: int, batch_size: Optional[int], options: Optional[int], read_concern: ReadConcern, collation: Optional[Mapping[str, Any]]=None, session: Optional[ClientSession]=None, allow_disk_use: Optional[bool]=None) -> SON[str, Any]:
    """Generate a find command document."""
    cmd: SON[str, Any] = SON([('find', coll)])
    if '$query' in spec:
        cmd.update([(_MODIFIERS[key], val) if key in _MODIFIERS else (key, val) for key, val in spec.items()])
        if '$explain' in cmd:
            cmd.pop('$explain')
        if '$readPreference' in cmd:
            cmd.pop('$readPreference')
    else:
        cmd['filter'] = spec
    if projection:
        cmd['projection'] = projection
    if skip:
        cmd['skip'] = skip
    if limit:
        cmd['limit'] = abs(limit)
        if limit < 0:
            cmd['singleBatch'] = True
    if batch_size:
        cmd['batchSize'] = batch_size
    if read_concern.level and (not (session and session.in_transaction)):
        cmd['readConcern'] = read_concern.document
    if collation:
        cmd['collation'] = collation
    if allow_disk_use is not None:
        cmd['allowDiskUse'] = allow_disk_use
    if options:
        cmd.update([(opt, True) for opt, val in _OPTIONS.items() if options & val])
    return cmd