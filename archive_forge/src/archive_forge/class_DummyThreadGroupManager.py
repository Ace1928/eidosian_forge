import functools
from heat.common import template_format
from heat.engine.clients.os import glance
from heat.engine.clients.os import keystone
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine.clients.os import nova
from heat.engine import environment
from heat.engine.resources.aws.ec2 import instance as instances
from heat.engine import stack as parser
from heat.engine import template as templatem
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
class DummyThreadGroupManager(object):

    def __init__(self):
        self.msg_queues = []
        self.messages = []

    def start(self, stack, func, *args, **kwargs):
        func(*args, **kwargs)
        return DummyThread()

    def start_with_lock(self, cnxt, stack, engine_id, func, *args, **kwargs):
        func(*args, **kwargs)
        return DummyThread()

    def start_with_acquired_lock(self, stack, lock, func, *args, **kwargs):
        func(*args, **kwargs)
        return DummyThread()

    def send(self, stack_id, message):
        self.messages.append(message)

    def add_msg_queue(self, stack_id, msg_queue):
        self.msg_queues.append(msg_queue)

    def remove_msg_queue(self, gt, stack_id, msg_queue):
        for q in self.msg_queues.pop(stack_id, []):
            if q is not msg_queue:
                self.add_event(stack_id, q)