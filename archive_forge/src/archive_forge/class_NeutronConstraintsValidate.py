from unittest import mock
from neutronclient.common import exceptions as qe
from heat.common import exception
from heat.engine.clients.os import neutron
from heat.engine.clients.os.neutron import lbaas_constraints as lc
from heat.engine.clients.os.neutron import neutron_constraints as nc
from heat.tests import common
from heat.tests import utils
class NeutronConstraintsValidate(common.HeatTestCase):
    scenarios = [('validate_network', dict(constraint_class=nc.NetworkConstraint, resource_type='network')), ('validate_port', dict(constraint_class=nc.PortConstraint, resource_type='port')), ('validate_router', dict(constraint_class=nc.RouterConstraint, resource_type='router')), ('validate_subnet', dict(constraint_class=nc.SubnetConstraint, resource_type='subnet')), ('validate_subnetpool', dict(constraint_class=nc.SubnetPoolConstraint, resource_type='subnetpool')), ('validate_address_scope', dict(constraint_class=nc.AddressScopeConstraint, resource_type='address_scope')), ('validate_loadbalancer', dict(constraint_class=lc.LoadbalancerConstraint, resource_type='loadbalancer')), ('validate_listener', dict(constraint_class=lc.ListenerConstraint, resource_type='listener')), ('validate_pool', dict(constraint_class=lc.PoolConstraint, resource_type='pool')), ('validate_qos_policy', dict(constraint_class=nc.QoSPolicyConstraint, resource_type='policy')), ('validate_security_group', dict(constraint_class=nc.SecurityGroupConstraint, resource_type='security_group'))]

    def test_validate(self):
        mock_extension = self.patchobject(neutron.NeutronClientPlugin, 'has_extension', return_value=True)
        nc = mock.Mock()
        mock_create = self.patchobject(neutron.NeutronClientPlugin, '_create')
        mock_create.return_value = nc
        mock_find = self.patchobject(neutron.NeutronClientPlugin, 'find_resourceid_by_name_or_id')
        mock_find.side_effect = ['foo', qe.NeutronClientException(status_code=404)]
        constraint = self.constraint_class()
        ctx = utils.dummy_context()
        if hasattr(constraint, 'extension') and constraint.extension:
            mock_extension.side_effect = [False, True, True]
            ex = self.assertRaises(exception.EntityNotFound, constraint.validate_with_client, ctx.clients, 'foo')
            expected = 'The neutron extension (%s) could not be found.' % constraint.extension
            self.assertEqual(expected, str(ex))
        self.assertTrue(constraint.validate('foo', ctx))
        self.assertFalse(constraint.validate('bar', ctx))
        mock_find.assert_has_calls([mock.call(self.resource_type, 'foo'), mock.call(self.resource_type, 'bar')])