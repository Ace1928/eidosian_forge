import struct
import sys
import time
def _write_pcap_file_hdr(self):
    pcap_file_hdr = PcapFileHdr(snaplen=self.snaplen, network=self.network)
    self._f.write(pcap_file_hdr.serialize())