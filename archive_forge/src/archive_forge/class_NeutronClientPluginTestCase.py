from unittest import mock
from neutronclient.common import exceptions as qe
from heat.common import exception
from heat.engine.clients.os import neutron
from heat.engine.clients.os.neutron import lbaas_constraints as lc
from heat.engine.clients.os.neutron import neutron_constraints as nc
from heat.tests import common
from heat.tests import utils
class NeutronClientPluginTestCase(common.HeatTestCase):

    def setUp(self):
        super(NeutronClientPluginTestCase, self).setUp()
        self.neutron_client = mock.MagicMock()
        con = utils.dummy_context()
        c = con.clients
        self.neutron_plugin = c.client_plugin('neutron')
        self.neutron_plugin.client = lambda: self.neutron_client