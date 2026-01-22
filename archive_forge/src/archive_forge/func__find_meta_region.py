import struct
from oslo_log import log as logging
def _find_meta_region(self):
    region_entry_first = 16
    regi, cksum, count, reserved = struct.unpack('<IIII', self.region('header').data[:16])
    if regi != 1768383858:
        raise ImageFormatError('Region signature not found at %x' % self.region('header').offset)
    if count >= 2048:
        raise ImageFormatError('Region count is %i (limit 2047)' % count)
    self._log.debug('Region entry first is %x', region_entry_first)
    self._log.debug('Region entries %i', count)
    meta_offset = 0
    for i in range(0, count):
        entry_start = region_entry_first + i * 32
        entry_end = entry_start + 32
        entry = self.region('header').data[entry_start:entry_end]
        self._log.debug('Entry offset is %x', entry_start)
        guid = self._guid(entry[:16])
        if guid == self.METAREGION:
            meta_offset, meta_len, meta_req = struct.unpack('<QII', entry[16:])
            self._log.debug('Meta entry %i specifies offset: %x', i, meta_offset)
            meta_len = 2048 * 32
            return CaptureRegion(meta_offset, meta_len)
    self._log.warning('Did not find metadata region')
    return None