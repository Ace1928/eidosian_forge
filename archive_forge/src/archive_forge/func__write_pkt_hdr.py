import struct
import sys
import time
def _write_pkt_hdr(self, ts, buf_len):
    sec = int(ts)
    usec = int(round(ts % 1, 6) * 1000000.0) if sec != 0 else 0
    pc_pkt_hdr = PcapPktHdr(ts_sec=sec, ts_usec=usec, incl_len=buf_len, orig_len=buf_len)
    self._f.write(pc_pkt_hdr.serialize())