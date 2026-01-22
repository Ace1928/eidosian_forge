import logging
import unittest
import os_ken.ofproto.ofproto_v1_5 as ofp
def _test_encode_header(self, user, on_wire):
    """ test encording header.

        t: oxs_type
        """
    t = ofp.oxs_from_user_header(user)
    buf = bytearray()
    ofp.oxs_serialize_header(t, buf, 0)
    self.assertEqual(on_wire, buf)