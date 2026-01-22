import os
from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
from openstack.tests.functional import base
class TestDevstack(base.BaseFunctionalTest):
    scenarios = [('designate', dict(env='DESIGNATE', service='dns')), ('heat', dict(env='HEAT', service='orchestration')), ('magnum', dict(env='MAGNUM', service='container-infrastructure-management')), ('neutron', dict(env='NEUTRON', service='network')), ('octavia', dict(env='OCTAVIA', service='load-balancer')), ('swift', dict(env='SWIFT', service='object-store'))]

    def test_has_service(self):
        if os.environ.get('OPENSTACKSDK_HAS_{env}'.format(env=self.env), '0') == '1':
            self.assertTrue(self.user_cloud.has_service(self.service))