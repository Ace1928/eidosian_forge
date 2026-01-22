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
class BaseTTXConverter(DefaultTable):
    """Generic base class for TTX table converters. It functions as an
    adapter between the TTX (ttLib actually) table model and the model
    we use for OpenType tables, which is necessarily subtly different.
    """

    def decompile(self, data, font):
        """Create an object from the binary data. Called automatically on access."""
        from . import otTables
        reader = OTTableReader(data, tableTag=self.tableTag)
        tableClass = getattr(otTables, self.tableTag)
        self.table = tableClass()
        self.table.decompile(reader, font)

    def compile(self, font):
        """Compiles the table into binary. Called automatically on save."""
        overflowRecord = None
        use_hb_repack = font.cfg[USE_HARFBUZZ_REPACKER]
        if self.tableTag in ('GSUB', 'GPOS'):
            if use_hb_repack is False:
                log.debug("hb.repack disabled, compiling '%s' with pure-python serializer", self.tableTag)
            elif not have_uharfbuzz:
                if use_hb_repack is True:
                    raise ImportError("No module named 'uharfbuzz'")
                else:
                    assert use_hb_repack is None
                    log.debug("uharfbuzz not found, compiling '%s' with pure-python serializer", self.tableTag)
        if use_hb_repack in (None, True) and have_uharfbuzz and (self.tableTag in ('GSUB', 'GPOS')):
            state = RepackerState.HB_FT
        else:
            state = RepackerState.PURE_FT
        hb_first_error_logged = False
        lastOverflowRecord = None
        while True:
            try:
                writer = OTTableWriter(tableTag=self.tableTag)
                self.table.compile(writer, font)
                if state == RepackerState.HB_FT:
                    return self.tryPackingHarfbuzz(writer, hb_first_error_logged)
                elif state == RepackerState.PURE_FT:
                    return self.tryPackingFontTools(writer)
                elif state == RepackerState.FT_FALLBACK:
                    self.tryPackingFontTools(writer)
                    log.debug('Re-enabling sharing between extensions and switching back to harfbuzz+fontTools packing.')
                    state = RepackerState.HB_FT
            except OTLOffsetOverflowError as e:
                hb_first_error_logged = True
                ok = self.tryResolveOverflow(font, e, lastOverflowRecord)
                lastOverflowRecord = e.value
                if ok:
                    continue
                if state is RepackerState.HB_FT:
                    log.debug('Harfbuzz packing out of resolutions, disabling sharing between extensions and switching to fontTools only packing.')
                    state = RepackerState.FT_FALLBACK
                else:
                    raise

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

    def tryPackingFontTools(self, writer):
        return writer.getAllData()

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

    def toXML(self, writer, font):
        self.table.toXML2(writer, font)

    def fromXML(self, name, attrs, content, font):
        from . import otTables
        if not hasattr(self, 'table'):
            tableClass = getattr(otTables, self.tableTag)
            self.table = tableClass()
        self.table.fromXML(name, attrs, content, font)
        self.table.populateDefaults()

    def ensureDecompiled(self, recurse=True):
        self.table.ensureDecompiled(recurse=recurse)