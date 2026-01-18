from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.resources.openstack.manila import share_network
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_update_security_services(self):
    net = self._create_network('share_network', self.rsrc_defn, self.stack)
    props = self.tmpl['resources']['share_network']['properties'].copy()
    props['security_services'] = ['7', '8']
    update_template = net.t.freeze(properties=props)
    scheduler.TaskRunner(net.update, update_template)()
    self.assertEqual((net.UPDATE, net.COMPLETE), net.state)
    called = net.client().share_networks.update.called
    self.assertFalse(called)
    net.client().share_networks.add_security_service.assert_called_with('42', '8')
    net.client().share_networks.remove_security_service.assert_called_with('42', '6')