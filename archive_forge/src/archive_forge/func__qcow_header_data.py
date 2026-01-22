import struct
from oslo_log import log as logging
def _qcow_header_data(self):
    magic, version, bf_offset, bf_sz, cluster_bits, size = struct.unpack('>4sIQIIQ', self.region('header').data[:32])
    return (magic, size)