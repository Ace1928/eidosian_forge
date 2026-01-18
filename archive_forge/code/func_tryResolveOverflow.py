from fontTools.config import OPTIONS
from fontTools.misc.textTools import Tag, bytesjoin
from .DefaultTable import DefaultTable
from enum import IntEnum
import sys
import array
import struct
import logging
from functools import lru_cache
from typing import Iterator, NamedTuple, Optional, Tuple
def tryResolveOverflow(self, font, e, lastOverflowRecord):
    ok = 0
    if lastOverflowRecord == e.value:
        return ok
    overflowRecord = e.value
    log.info('Attempting to fix OTLOffsetOverflowError %s', e)
    if overflowRecord.itemName is None:
        from .otTables import fixLookupOverFlows
        ok = fixLookupOverFlows(font, overflowRecord)
    else:
        from .otTables import fixSubTableOverFlows
        ok = fixSubTableOverFlows(font, overflowRecord)
    if ok:
        return ok
    from .otTables import fixLookupOverFlows
    return fixLookupOverFlows(font, overflowRecord)