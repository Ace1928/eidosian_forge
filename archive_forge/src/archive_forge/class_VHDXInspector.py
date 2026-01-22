import struct
from oslo_log import log as logging
class VHDXInspector(FileInspector):
    """MS VHDX Format

    This requires some complex parsing of the stream. The first 256KiB
    of the image is stored to get the header and region information,
    and then we capture the first metadata region to read those
    records, find the location of the virtual size data and parse
    it. This needs to store the metadata table entries up until the
    VDS record, which may consist of up to 2047 32-byte entries at
    max.  Finally, it must store a chunk of data at the offset of the
    actual VDS uint64.

    """
    METAREGION = '8B7CA206-4790-4B9A-B8FE-575F050F886E'
    VIRTUAL_DISK_SIZE = '2FA54224-CD1B-4876-B211-5DBED83BF4B8'
    VHDX_METADATA_TABLE_MAX_SIZE = 32 * 2048

    def __init__(self, *a, **k):
        super(VHDXInspector, self).__init__(*a, **k)
        self.new_region('ident', CaptureRegion(0, 32))
        self.new_region('header', CaptureRegion(192 * 1024, 64 * 1024))

    def post_process(self):
        if self.region('header').complete and (not self.has_region('metadata')):
            region = self._find_meta_region()
            if region:
                self.new_region('metadata', region)
        elif self.has_region('metadata') and (not self.has_region('vds')):
            region = self._find_meta_entry(self.VIRTUAL_DISK_SIZE)
            if region:
                self.new_region('vds', region)

    @property
    def format_match(self):
        return self.region('ident').data.startswith(b'vhdxfile')

    @staticmethod
    def _guid(buf):
        """Format a MSFT GUID from the 16-byte input buffer."""
        guid_format = '<IHHBBBBBBBB'
        return '%08X-%04X-%04X-%02X%02X-%02X%02X%02X%02X%02X%02X' % struct.unpack(guid_format, buf)

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

    def _find_meta_entry(self, desired_guid):
        meta_buffer = self.region('metadata').data
        if len(meta_buffer) < 32:
            return None
        sig, reserved, count = struct.unpack('<8sHH', meta_buffer[:12])
        if sig != b'metadata':
            raise ImageFormatError('Invalid signature for metadata region: %r' % sig)
        entries_size = 32 + count * 32
        if len(meta_buffer) < entries_size:
            return None
        if count >= 2048:
            raise ImageFormatError('Metadata item count is %i (limit 2047)' % count)
        for i in range(0, count):
            entry_offset = 32 + i * 32
            guid = self._guid(meta_buffer[entry_offset:entry_offset + 16])
            if guid == desired_guid:
                item_offset, item_length, _reserved = struct.unpack('<III', meta_buffer[entry_offset + 16:entry_offset + 28])
                item_length = min(item_length, self.VHDX_METADATA_TABLE_MAX_SIZE)
                self.region('metadata').length = len(meta_buffer)
                self._log.debug('Found entry at offset %x', item_offset)
                return CaptureRegion(self.region('metadata').offset + item_offset, item_length)
        self._log.warning('Did not find guid %s', desired_guid)
        return None

    @property
    def virtual_size(self):
        if not self.has_region('vds') or not self.region('vds').complete:
            return 0
        size, = struct.unpack('<Q', self.region('vds').data)
        return size

    def __str__(self):
        return 'vhdx'