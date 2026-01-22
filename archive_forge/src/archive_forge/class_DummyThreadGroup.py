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
class DummyThreadGroup(object):

    def __init__(self):
        self.threads = []

    def add_timer(self, interval, callback, initial_delay=None, *args, **kwargs):
        self.threads.append(callback)

    def stop_timers(self):
        pass

    def add_thread(self, callback, cnxt, trace, func, *args, **kwargs):
        self.threads.append(func)
        return DummyThread()

    def stop(self, graceful=False):
        pass

    def wait(self):
        pass