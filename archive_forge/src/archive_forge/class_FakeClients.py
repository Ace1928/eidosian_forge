import collections
import json
from unittest import mock
from heatclient import exc
from heatclient.v1 import stacks
from keystoneauth1 import loading as ks_loading
from oslo_config import cfg
from heat.common import exception
from heat.common.i18n import _
from heat.common import policy
from heat.common import template_format
from heat.engine.clients.os import heat_plugin
from heat.engine import environment
from heat.engine import node_data
from heat.engine import resource
from heat.engine.resources.openstack.heat import remote_stack
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack
from heat.engine import template
from heat.tests import common as tests_common
from heat.tests import utils
class FakeClients(object):

    def __init__(self, context, region_name=None):
        self.ctx = context
        self.region_name = region_name or 'RegionOne'
        self.hc = None
        self.plugin = None

    def client(self, name):
        if self.region_name in ['RegionOne', 'RegionTwo']:
            if self.hc is None:
                self.hc = mock.MagicMock()
            return self.hc
        else:
            raise Exception('Failed connecting to Heat')

    def client_plugin(self, name):
        if self.plugin is None:
            self.plugin = heat_plugin.HeatClientPlugin(self.ctx)
        return self.plugin