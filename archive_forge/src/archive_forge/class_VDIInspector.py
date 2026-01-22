import struct
from oslo_log import log as logging
class VDIInspector(FileInspector):
    """VirtualBox VDI format

    This only needs to store the first 512 bytes of the image.
    """

    def __init__(self, *a, **k):
        super(VDIInspector, self).__init__(*a, **k)
        self.new_region('header', CaptureRegion(0, 512))

    @property
    def format_match(self):
        if not self.region('header').complete:
            return False
        signature, = struct.unpack('<I', self.region('header').data[64:68])
        return signature == 3201962111

    @property
    def virtual_size(self):
        if not self.region('header').complete:
            return 0
        if not self.format_match:
            return 0
        size, = struct.unpack('<Q', self.region('header').data[368:376])
        return size

    def __str__(self):
        return 'vdi'