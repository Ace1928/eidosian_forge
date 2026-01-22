import unittest
import logging
import socket
from struct import *
from os_ken.ofproto.ofproto_v1_2_parser import *
from os_ken.ofproto import ofproto_v1_2_parser
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_protocol
from os_ken.ofproto import ether
from os_ken.ofproto.ofproto_parser import MsgBase
from os_ken import utils
from os_ken.lib import addrconv
from os_ken.lib import pack_utils
class TestOFPFlowMod(unittest.TestCase):
    """ Test case for ofproto_v1_2_parser.OFPFlowMod
    """

    def test_init(self):
        cookie = 2127614848199081640
        cookie_mask = 2127614848199081641
        table_id = 3
        command = 0
        idle_timeout = 62317
        hard_timeout = 7365
        priority = 40163
        buffer_id = 4037115955
        out_port = 65037
        out_group = 6606
        flags = 135
        instructions = [OFPInstructionGotoTable(table_id)]
        in_port = 1
        match = OFPMatch()
        match.set_in_port(in_port)
        c = OFPFlowMod(_Datapath, cookie, cookie_mask, table_id, command, idle_timeout, hard_timeout, priority, buffer_id, out_port, out_group, flags, match, instructions)
        self.assertEqual(cookie, c.cookie)
        self.assertEqual(cookie_mask, c.cookie_mask)
        self.assertEqual(table_id, c.table_id)
        self.assertEqual(command, c.command)
        self.assertEqual(idle_timeout, c.idle_timeout)
        self.assertEqual(hard_timeout, c.hard_timeout)
        self.assertEqual(priority, c.priority)
        self.assertEqual(buffer_id, c.buffer_id)
        self.assertEqual(out_port, c.out_port)
        self.assertEqual(out_group, c.out_group)
        self.assertEqual(flags, c.flags)
        self.assertEqual(in_port, c.match._flow.in_port)
        self.assertEqual(instructions[0], c.instructions[0])

    def _test_serialize(self, cookie, cookie_mask, table_id, command, idle_timeout, hard_timeout, priority, buffer_id, out_port, out_group, flags, inst_cnt=0):
        dl_type = 2048
        match = OFPMatch()
        match.set_dl_type(dl_type)
        insts = []
        for i in range(inst_cnt):
            insts.append(OFPInstructionGotoTable(i))
        c = OFPFlowMod(_Datapath, cookie, cookie_mask, table_id, command, idle_timeout, hard_timeout, priority, buffer_id, out_port, out_group, flags, match, insts)
        c.serialize()
        self.assertEqual(ofproto.OFP_VERSION, c.version)
        self.assertEqual(ofproto.OFPT_FLOW_MOD, c.msg_type)
        self.assertEqual(0, c.xid)
        fmt = ofproto.OFP_HEADER_PACK_STR + ofproto.OFP_FLOW_MOD_PACK_STR0[1:] + 'HHHBB' + MTEthType.pack_str[1:] + '6x' + ofproto.OFP_INSTRUCTION_GOTO_TABLE_PACK_STR[1:] * inst_cnt
        res = struct.unpack(fmt, bytes(c.buf))
        self.assertEqual(res[0], ofproto.OFP_VERSION)
        self.assertEqual(res[1], ofproto.OFPT_FLOW_MOD)
        self.assertEqual(res[2], len(c.buf))
        self.assertEqual(res[3], 0)
        self.assertEqual(res[4], cookie)
        self.assertEqual(res[5], cookie_mask)
        self.assertEqual(res[6], table_id)
        self.assertEqual(res[7], command)
        self.assertEqual(res[8], idle_timeout)
        self.assertEqual(res[9], hard_timeout)
        self.assertEqual(res[10], priority)
        self.assertEqual(res[11], buffer_id)
        self.assertEqual(res[12], out_port)
        self.assertEqual(res[13], out_group)
        self.assertEqual(res[14], flags)
        self.assertEqual(res[15], ofproto.OFPMT_OXM)
        self.assertEqual(res[16], 10)
        self.assertEqual(res[17], ofproto.OFPXMC_OPENFLOW_BASIC)
        self.assertEqual(res[18] >> 1, ofproto.OFPXMT_OFB_ETH_TYPE)
        self.assertEqual(res[18] & 1, 0)
        self.assertEqual(res[19], calcsize(MTEthType.pack_str))
        self.assertEqual(res[20], dl_type)
        for i in range(inst_cnt):
            index = 21 + 3 * i
            self.assertEqual(res[index], ofproto.OFPIT_GOTO_TABLE)
            self.assertEqual(res[index + 1], ofproto.OFP_INSTRUCTION_GOTO_TABLE_SIZE)
            self.assertEqual(res[index + 2], i)

    def test_serialize_mid(self):
        cookie = 2127614848199081640
        cookie_mask = 2127614848199081641
        table_id = 3
        command = 128
        idle_timeout = 62317
        hard_timeout = 7365
        priority = 40163
        buffer_id = 4037115955
        out_port = 65037
        out_group = 6606
        flags = 135
        inst_cnt = 1
        self._test_serialize(cookie, cookie_mask, table_id, command, idle_timeout, hard_timeout, priority, buffer_id, out_port, out_group, flags, inst_cnt)

    def test_serialize_max(self):
        cookie = 18446744073709551615
        cookie_mask = 18446744073709551615
        table_id = 255
        command = 255
        idle_timeout = 65535
        hard_timeout = 65535
        priority = 65535
        buffer_id = 4294967295
        out_port = 4294967295
        out_group = 4294967295
        flags = 65535
        inst_cnt = 255
        self._test_serialize(cookie, cookie_mask, table_id, command, idle_timeout, hard_timeout, priority, buffer_id, out_port, out_group, flags, inst_cnt)

    def test_serialize_min(self):
        cookie = 0
        cookie_mask = 0
        table_id = 0
        command = ofproto.OFPFC_ADD
        idle_timeout = 0
        hard_timeout = 0
        priority = 0
        buffer_id = 0
        out_port = 0
        out_group = 0
        flags = 0
        self._test_serialize(cookie, cookie_mask, table_id, command, idle_timeout, hard_timeout, priority, buffer_id, out_port, out_group, flags)

    def test_serialize_p1(self):
        cookie = 2127614848199081640
        cookie_mask = 2127614848199081641
        table_id = 3
        command = 1
        idle_timeout = 62317
        hard_timeout = 7365
        priority = 40163
        buffer_id = 4037115955
        out_port = 65037
        out_group = 6606
        flags = 1 << 0
        self._test_serialize(cookie, cookie_mask, table_id, command, idle_timeout, hard_timeout, priority, buffer_id, out_port, out_group, flags)

    def test_serialize_p2(self):
        cookie = 2127614848199081640
        cookie_mask = 2127614848199081641
        table_id = 3
        command = 2
        idle_timeout = 62317
        hard_timeout = 7365
        priority = 40163
        buffer_id = 4037115955
        out_port = 65037
        out_group = 6606
        flags = 1 << 0
        self._test_serialize(cookie, cookie_mask, table_id, command, idle_timeout, hard_timeout, priority, buffer_id, out_port, out_group, flags)

    def test_serialize_p3(self):
        cookie = 2127614848199081640
        cookie_mask = 2127614848199081641
        table_id = 3
        command = 3
        idle_timeout = 62317
        hard_timeout = 7365
        priority = 40163
        buffer_id = 4037115955
        out_port = 65037
        out_group = 6606
        flags = 1 << 1
        self._test_serialize(cookie, cookie_mask, table_id, command, idle_timeout, hard_timeout, priority, buffer_id, out_port, out_group, flags)

    def test_serialize_p4(self):
        cookie = 2127614848199081640
        cookie_mask = 2127614848199081641
        table_id = 3
        command = 4
        idle_timeout = 62317
        hard_timeout = 7365
        priority = 40163
        buffer_id = 4037115955
        out_port = 65037
        out_group = 6606
        flags = 1 << 2
        self._test_serialize(cookie, cookie_mask, table_id, command, idle_timeout, hard_timeout, priority, buffer_id, out_port, out_group, flags)