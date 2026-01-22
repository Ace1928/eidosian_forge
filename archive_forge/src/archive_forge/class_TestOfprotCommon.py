from importlib import reload
import unittest
import logging
class TestOfprotCommon(unittest.TestCase):
    """ Test case for ofproto
    """

    def test_ofp_event(self):
        import os_ken.ofproto
        reload(os_ken.ofproto)
        import os_ken.controller.ofp_event
        reload(os_ken.controller.ofp_event)

    def test_ofproto(self):
        import os_ken.ofproto
        reload(os_ken.ofproto)
        ofp_modules = os_ken.ofproto.get_ofp_modules()
        import os_ken.ofproto.ofproto_v1_0
        import os_ken.ofproto.ofproto_v1_2
        import os_ken.ofproto.ofproto_v1_3
        import os_ken.ofproto.ofproto_v1_4
        import os_ken.ofproto.ofproto_v1_5
        self.assertEqual(set(ofp_modules.keys()), set([os_ken.ofproto.ofproto_v1_0.OFP_VERSION, os_ken.ofproto.ofproto_v1_2.OFP_VERSION, os_ken.ofproto.ofproto_v1_3.OFP_VERSION, os_ken.ofproto.ofproto_v1_4.OFP_VERSION, os_ken.ofproto.ofproto_v1_5.OFP_VERSION]))
        consts_mods = set([ofp_mod[0] for ofp_mod in ofp_modules.values()])
        self.assertEqual(consts_mods, set([os_ken.ofproto.ofproto_v1_0, os_ken.ofproto.ofproto_v1_2, os_ken.ofproto.ofproto_v1_3, os_ken.ofproto.ofproto_v1_4, os_ken.ofproto.ofproto_v1_5]))
        parser_mods = set([ofp_mod[1] for ofp_mod in ofp_modules.values()])
        import os_ken.ofproto.ofproto_v1_0_parser
        import os_ken.ofproto.ofproto_v1_2_parser
        import os_ken.ofproto.ofproto_v1_3_parser
        import os_ken.ofproto.ofproto_v1_4_parser
        import os_ken.ofproto.ofproto_v1_5_parser
        self.assertEqual(parser_mods, set([os_ken.ofproto.ofproto_v1_0_parser, os_ken.ofproto.ofproto_v1_2_parser, os_ken.ofproto.ofproto_v1_3_parser, os_ken.ofproto.ofproto_v1_4_parser, os_ken.ofproto.ofproto_v1_5_parser]))