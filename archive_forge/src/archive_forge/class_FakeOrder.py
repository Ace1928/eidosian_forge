from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.resources.openstack.barbican import order
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
class FakeOrder(object):

    def __init__(self, name):
        self.name = name

    def submit(self):
        return self.name