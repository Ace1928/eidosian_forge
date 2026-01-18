import struct
from oslo_log import log as logging
@property
def virtual_size(self):
    if not self.region('header').complete:
        return 0
    if not self.format_match:
        return 0
    size, = struct.unpack('<Q', self.region('header').data[368:376])
    return size