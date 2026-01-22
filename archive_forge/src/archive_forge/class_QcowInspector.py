import struct
from oslo_log import log as logging
class QcowInspector(FileInspector):
    """QEMU QCOW2 Format

    This should only require about 32 bytes of the beginning of the file
    to determine the virtual size.
    """

    def __init__(self, *a, **k):
        super(QcowInspector, self).__init__(*a, **k)
        self.new_region('header', CaptureRegion(0, 512))

    def _qcow_header_data(self):
        magic, version, bf_offset, bf_sz, cluster_bits, size = struct.unpack('>4sIQIIQ', self.region('header').data[:32])
        return (magic, size)

    @property
    def virtual_size(self):
        if not self.region('header').complete:
            return 0
        if not self.format_match:
            return 0
        magic, size = self._qcow_header_data()
        return size

    @property
    def format_match(self):
        if not self.region('header').complete:
            return False
        magic, size = self._qcow_header_data()
        return magic == b'QFI\xfb'

    def __str__(self):
        return 'qcow2'