import logging
import unittest
import os_ken.ofproto.ofproto_v1_5 as ofp
def _test_decode_header(self, user, on_wire):
    """ test decording header.

        t: oxs_type
        l: length of header
        n: name of OXS field
        """
    t, l = ofp.oxs_parse_header(on_wire, 0)
    self.assertEqual(len(on_wire), l)
    n = ofp.oxs_to_user_header(t)
    self.assertEqual(user, n)