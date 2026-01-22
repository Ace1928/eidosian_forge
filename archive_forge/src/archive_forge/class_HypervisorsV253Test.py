from novaclient import api_versions
from novaclient import exceptions
from novaclient.tests.unit.fixture_data import client
from novaclient.tests.unit.fixture_data import hypervisors as data
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
class HypervisorsV253Test(HypervisorsV233Test):
    """Tests the os-hypervisors 2.53 API bindings."""
    data_fixture_class = data.V253

    def setUp(self):
        super(HypervisorsV253Test, self).setUp()
        self.cs.api_version = api_versions.APIVersion('2.53')

    def test_hypervisor_search_detailed(self):
        expected = [dict(id=self.data_fixture.hyper_id_1, state='up', status='enabled', hypervisor_hostname='hyper1'), dict(id=self.data_fixture.hyper_id_2, state='up', status='enabled', hypervisor_hostname='hyper2')]
        result = self.cs.hypervisors.search('hyper', detailed=True)
        self.assert_request_id(result, fakes.FAKE_REQUEST_ID_LIST)
        self.assert_called('GET', '/os-hypervisors/detail?hypervisor_hostname_pattern=hyper')
        for idx, hyper in enumerate(result):
            self.compare_to_expected(expected[idx], hyper)