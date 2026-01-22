from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.resources.openstack.manila import share_network
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
class DummyShareNetwork(object):

    def __init__(self):
        self.id = '42'
        self.segmentation_id = '2'
        self.cidr = '3'
        self.ip_version = '5'
        self.network_type = '6'