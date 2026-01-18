import json
import os
import ssl
import sys
import warnings
import logging
import random
import testtools
import unittest
from unittest import mock
from os_ken.base import app_manager  # To suppress cyclic import
from os_ken.controller import controller
from os_ken.controller import handler
from os_ken.lib import hub
from os_ken.ofproto import ofproto_v1_3_parser
from os_ken.ofproto import ofproto_v1_2_parser
from os_ken.ofproto import ofproto_v1_0_parser
@mock.patch('os_ken.base.app_manager', spec=app_manager)
def test_recv_loop(self, app_manager_mock):
    test_messages = ['4-6-ofp_features_reply.packet', '4-14-ofp_echo_reply.packet', '4-14-ofp_echo_reply.packet', '4-4-ofp_packet_in.packet', '4-14-ofp_echo_reply.packet', '4-14-ofp_echo_reply.packet']
    this_dir = os.path.dirname(sys.modules[__name__].__file__)
    packet_data_dir = os.path.join(this_dir, '../../packet_data/of13')
    json_dir = os.path.join(this_dir, '../ofproto/json/of13')
    packet_buf = bytearray()
    expected_json = list()
    for msg in test_messages:
        packet_data_file = os.path.join(packet_data_dir, msg)
        packet_buf += open(packet_data_file, 'rb').read()
        json_data_file = os.path.join(json_dir, msg + '.json')
        expected_json.append(json.load(open(json_data_file)))

    class SocketMock(mock.MagicMock):
        buf = bytearray()
        random = None

        def recv(self, bufsize):
            size = self.random.randint(1, bufsize)
            out = self.buf[:size]
            self.buf = self.buf[size:]
            return out
    ofp_brick_mock = mock.MagicMock(spec=app_manager.OSKenApp)
    app_manager_mock.lookup_service_brick.return_value = ofp_brick_mock
    sock_mock = SocketMock()
    sock_mock.buf = packet_buf
    sock_mock.random = random.Random('OSKen SDN Framework')
    addr_mock = mock.MagicMock()
    dp = controller.Datapath(sock_mock, addr_mock)
    dp.set_state(handler.MAIN_DISPATCHER)
    ofp_brick_mock.reset_mock()
    dp._recv_loop()
    output_json = list()
    for call in ofp_brick_mock.send_event_to_observers.call_args_list:
        args, kwargs = call
        ev, state = args
        if not hasattr(ev, 'msg'):
            continue
        output_json.append(ev.msg.to_jsondict())
        self.assertEqual(state, handler.MAIN_DISPATCHER)
        self.assertEqual(kwargs, {})
    self.assertEqual(expected_json, output_json)