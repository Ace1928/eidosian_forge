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
def tryPackingHarfbuzz(self, writer, hb_first_error_logged):
    try:
        log.debug("serializing '%s' with hb.repack", self.tableTag)
        return writer.getAllDataUsingHarfbuzz(self.tableTag)
    except (ValueError, MemoryError, hb.RepackerError) as e:
        if not hb_first_error_logged:
            error_msg = f'{type(e).__name__}'
            if str(e) != '':
                error_msg += f': {e}'
            log.warning("hb.repack failed to serialize '%s', attempting fonttools resolutions ; the error message was: %s", self.tableTag, error_msg)
            hb_first_error_logged = True
        return writer.getAllData(remove_duplicate=False)