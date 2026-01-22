import json
from unittest import mock
from novaclient import exceptions
from oslo_utils import excutils
from heat.common import template_format
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
class FakeGroup(object):

    def __init__(self, name):
        self.id = name
        self.name = name