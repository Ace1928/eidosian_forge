import logging
import unittest
import os_ken.ofproto.ofproto_v1_5 as ofp
def _test_decode(self, user, on_wire):
    """ test decording user value from on-wire bytes.

        t: oxs_type
        v: on-wire bytes value
        l: length of field
        n: name of OXS field
        uv: user vale
        """
    t, v, _, l = ofp.oxs_parse(on_wire, 0)
    self.assertEqual(len(on_wire), l)
    n, uv = ofp.oxs_to_user(t, v, None)
    self.assertEqual(user, (n, uv))