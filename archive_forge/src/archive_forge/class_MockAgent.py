import unittest
import time
import uuid
from unittest import mock
from parlai.mturk.core.dev.socket_manager import Packet, SocketManager
from parlai.mturk.core.dev.agents import AssignState
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.shared_utils as shared_utils
import threading
from websocket_server import WebsocketServer
import json
class MockAgent(object):
    """
    Class that pretends to be an MTurk agent interacting through the webpage by
    simulating the same commands that are sent from the core.html file.

    Exposes methods to use for testing and checking status
    """

    def __init__(self, hit_id, assignment_id, worker_id, task_group_id):
        self.conversation_id = None
        self.id = None
        self.assignment_id = assignment_id
        self.hit_id = hit_id
        self.worker_id = worker_id
        self.some_agent_disconnected = False
        self.disconnected = False
        self.task_group_id = task_group_id
        self.ws = None
        self.ready = False
        self.wants_to_send = False

    def send_packet(self, packet):

        def callback(*args):
            pass
        event_name = data_model.MESSAGE_BATCH
        self.ws.send(json.dumps({'type': event_name, 'content': packet.as_dict()}))

    def register_to_socket(self, ws, on_msg):
        handler = self.make_packet_handler(on_msg)
        self.ws = ws
        self.ws.handlers[self.worker_id] = handler

    def make_packet_handler(self, on_msg):
        """
        A packet handler.
        """

        def handler_mock(pkt):
            if pkt['type'] == data_model.WORLD_MESSAGE:
                packet = Packet.from_dict(pkt)
                on_msg(packet)
            elif pkt['type'] == data_model.MESSAGE_BATCH:
                packet = Packet.from_dict(pkt)
                on_msg(packet)
            elif pkt['type'] == data_model.AGENT_ALIVE:
                raise Exception('Invalid alive packet {}'.format(pkt))
            else:
                raise Exception('Invalid Packet type {} received in {}'.format(pkt['type'], pkt))
        return handler_mock

    def build_and_send_packet(self, packet_type, data):
        msg_id = str(uuid.uuid4())
        msg = {'id': msg_id, 'type': packet_type, 'sender_id': self.worker_id, 'assignment_id': self.assignment_id, 'conversation_id': self.conversation_id, 'receiver_id': '[World_' + self.task_group_id + ']', 'data': data}
        if packet_type == data_model.MESSAGE_BATCH:
            msg['data'] = {'messages': [{'id': msg_id, 'type': packet_type, 'sender_id': self.worker_id, 'assignment_id': self.assignment_id, 'conversation_id': self.conversation_id, 'receiver_id': '[World_' + self.task_group_id + ']', 'data': data}]}
        self.ws.send(json.dumps({'type': packet_type, 'content': msg}))
        return msg['id']

    def send_message(self, text):
        data = {'text': text, 'id': self.id, 'message_id': str(uuid.uuid4()), 'episode_done': False}
        self.wants_to_send = False
        return self.build_and_send_packet(data_model.MESSAGE_BATCH, data)

    def send_disconnect(self):
        data = {'hit_id': self.hit_id, 'assignment_id': self.assignment_id, 'worker_id': self.worker_id, 'conversation_id': self.conversation_id, 'connection_id': '{}_{}'.format(self.worker_id, self.assignment_id)}
        return self.build_and_send_packet(data_model.AGENT_DISCONNECT, data)

    def send_alive(self):
        data = {'hit_id': self.hit_id, 'assignment_id': self.assignment_id, 'worker_id': self.worker_id, 'conversation_id': self.conversation_id}
        return self.build_and_send_packet(data_model.AGENT_ALIVE, data)