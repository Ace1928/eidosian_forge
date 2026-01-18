import logging
import os
import sys
import unittest
from os_ken.utils import binary_str
from os_ken.lib import pcaplib
from os_ken.lib.packet import packet
from os_ken.lib.packet import bgp
from os_ken.lib.packet import afi
from os_ken.lib.packet import safi
def test_stream_parser(self):
    msgs = [bgp.BGPNotification(error_code=1, error_subcode=2, data=b'foo'), bgp.BGPNotification(error_code=3, error_subcode=4, data=b'bar'), bgp.BGPNotification(error_code=5, error_subcode=6, data=b'baz')]
    binmsgs = b''.join([bytes(msg.serialize()) for msg in msgs])
    sp = bgp.StreamParser()
    results = []
    for b in binmsgs:
        for m in sp.parse(b):
            results.append(m)
    self.assertEqual(str(results), str(msgs))