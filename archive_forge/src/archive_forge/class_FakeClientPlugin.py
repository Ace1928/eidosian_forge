import copy
from unittest import mock
from heat.common import exception
from heat.engine.cfn import functions as cfn_funcs
from heat.engine import function
from heat.engine.hot import functions as hot_funcs
from heat.engine import parameters
from heat.engine import properties
from heat.engine import translation
from heat.tests import common
class FakeClientPlugin(object):

    def find_name_id(self, entity=None, value=None):
        if entity == 'net':
            return 'net1_id'
        if entity == 'port':
            return 'port1_id'