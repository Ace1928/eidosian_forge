import logging
import unittest
import os_ken.ofproto.ofproto_v1_5 as ofp
def _test_encode(self, user, on_wire):
    """ test encording user value into on-wire bytes.

        n: name of OXS field
        uv: user vale
        t: oxs_type
        v: on-wire bytes value
        """
    n, uv = user
    t, v, _ = ofp.oxs_from_user(n, uv)
    buf = bytearray()
    ofp.oxs_serialize(t, v, None, buf, 0)
    self.assertEqual(on_wire, buf)