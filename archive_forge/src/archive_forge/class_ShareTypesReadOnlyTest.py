import ddt
from tempest.lib.common.utils import data_utils
from manilaclient import api_versions
from manilaclient.tests.functional import base
from manilaclient.tests.unit.v2 import test_types as unit_test_types
@ddt.ddt
class ShareTypesReadOnlyTest(base.BaseTestCase):

    @ddt.data(('admin', '1.0'), ('admin', '2.0'), ('admin', '2.6'), ('admin', '2.7'), ('user', '1.0'), ('user', '2.0'), ('user', '2.6'), ('user', '2.7'))
    @ddt.unpack
    def test_share_type_list(self, role, microversion):
        self.skip_if_microversion_not_supported(microversion)
        self.clients[role].manila('type-list', microversion=microversion)

    @ddt.data('1.0', '2.0', '2.6', '2.7')
    def test_extra_specs_list(self, microversion):
        self.skip_if_microversion_not_supported(microversion)
        self.admin_client.manila('extra-specs-list', microversion=microversion)