from unittest import mock
from ironicclient import exceptions as ic_exc
from heat.engine.clients.os import ironic as ic
from heat.tests import common
from heat.tests import utils
class fake_resource(object):

    def __init__(self, id=None, name=None):
        self.uuid = id
        self.name = name