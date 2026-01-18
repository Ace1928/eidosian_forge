import logging
import os
import sys
import unittest
from os_ken.lib import pcaplib
from os_ken.lib.packet import openflow
from os_ken.lib.packet import packet
from os_ken.utils import binary_str
def test_pcap(self):
    files = ['openflow_flowmod', 'openflow_flowstats_req', 'openflow_invalid_version']
    for f in files:
        for _, buf in pcaplib.Reader(open(OPENFLOW_DATA_DIR + f + '.pcap', 'rb')):
            pkt = packet.Packet(buf)
            openflow_pkt = pkt.get_protocol(openflow.openflow)
            self.assertTrue(isinstance(openflow_pkt, openflow.openflow), 'Failed to parse OpenFlow message: %s' % pkt)
            pkt.serialize()
            self.assertEqual(buf, pkt.data, "b'%s' != b'%s'" % (binary_str(buf), binary_str(pkt.data)))