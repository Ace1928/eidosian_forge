import unittest
import testscenarios
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_v1_3
from os_ken.ofproto import ofproto_v1_2_parser
from os_ken.ofproto import ofproto_v1_3_parser
from os_ken.lib import addrconv
from struct import unpack
class Test_Parser_Compat(testscenarios.WithScenarios, unittest.TestCase):
    scenarios = [(case['name'], case) for case in _list_test_cases()]

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_parser(self):
        self._test(name=self.name, ofpp=self.ofpp)

    def _test(self, name, ofpp):
        ofp = {ofproto_v1_2_parser: ofproto_v1_2, ofproto_v1_3_parser: ofproto_v1_3}[ofpp]
        in_port = 989284321
        eth_src = 'aa:bb:cc:dd:ee:ff'
        ipv4_src = '192.0.2.9'
        ipv6_src = 'fe80::f00b:a4ff:feef:5d8f'
        old_in_port = in_port
        old_eth_src = addrconv.mac.text_to_bin(eth_src)
        old_ipv4_src = unpack('!I', addrconv.ipv4.text_to_bin(ipv4_src))[0]
        old_ipv6_src = list(unpack('!8H', addrconv.ipv6.text_to_bin(ipv6_src)))

        def check(o):
            check_old(o)
            check_new(o)

        def check_old(o):

            def get_field(m, t):
                for f in m.fields:
                    if isinstance(f, t):
                        return f
            get_value = lambda m, t: get_field(m, t).value
            self.assertEqual(get_value(o, ofpp.MTInPort), old_in_port)
            self.assertEqual(get_value(o, ofpp.MTEthSrc), old_eth_src)
            self.assertEqual(get_value(o, ofpp.MTIPV4Src), old_ipv4_src)
            self.assertEqual(get_value(o, ofpp.MTIPv6Src), old_ipv6_src)

        def check_new(o):
            self.assertEqual(o['in_port'], in_port)
            self.assertEqual(o['eth_src'], eth_src)
            self.assertEqual(o['ipv4_src'], ipv4_src)
            self.assertEqual(o['ipv6_src'], ipv6_src)
        old = ofpp.OFPMatch()
        old.set_in_port(old_in_port)
        old.set_dl_src(old_eth_src)
        old.set_ipv4_src(old_ipv4_src)
        old.set_ipv6_src(old_ipv6_src)
        old_buf = bytearray()
        old.serialize(old_buf, 0)
        check_old(old)
        old2 = ofpp.OFPMatch()
        old2.append_field(ofp.OXM_OF_IN_PORT, old_in_port)
        old2.append_field(ofp.OXM_OF_ETH_SRC, old_eth_src)
        old2.append_field(ofp.OXM_OF_IPV4_SRC, old_ipv4_src)
        old2.append_field(ofp.OXM_OF_IPV6_SRC, old_ipv6_src)
        check_old(old2)
        old2_buf = bytearray()
        old2.serialize(old2_buf, 0)
        new = ofpp.OFPMatch(in_port=in_port, eth_src=eth_src, ipv4_src=ipv4_src, ipv6_src=ipv6_src)
        check_new(new)
        new_buf = bytearray()
        new.serialize(new_buf, 0)
        self.assertEqual(new_buf, old_buf)
        self.assertEqual(new_buf, old2_buf)
        old_jsondict = old.to_jsondict()
        old2_jsondict = old2.to_jsondict()
        new_jsondict = new.to_jsondict()
        self.assertEqual(new_jsondict, old_jsondict)
        self.assertEqual(new_jsondict, old2_jsondict)
        self.assertEqual(str(new), str(old))
        self.assertEqual(str(new), str(old2))
        check(ofpp.OFPMatch.parser(bytes(new_buf), 0))
        check(ofpp.OFPMatch.from_jsondict(list(new_jsondict.values())[0]))