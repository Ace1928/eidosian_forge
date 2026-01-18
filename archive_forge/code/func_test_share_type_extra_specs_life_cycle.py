import ddt
from tempest.lib.common.utils import data_utils
from manilaclient import api_versions
from manilaclient.tests.functional import base
from manilaclient.tests.unit.v2 import test_types as unit_test_types
@ddt.data((True, False), (True, True), (False, True), (False, False), (False, False, '2.6'), (False, False, '2.7'))
@ddt.unpack
def test_share_type_extra_specs_life_cycle(self, is_public, dhss, microversion=None):
    if microversion:
        self.skip_if_microversion_not_supported(microversion)
    st = self.create_share_type(driver_handles_share_servers=dhss, is_public=is_public, microversion=microversion)
    st_extra_specs = dict(foo_key='foo_value', bar_key='bar_value')
    self.admin_client.set_share_type_extra_specs(st['ID'], st_extra_specs, microversion=microversion)
    extra_specs = self.admin_client.list_share_type_extra_specs(st['ID'], microversion=microversion)
    for k, v in st_extra_specs.items():
        self.assertIn('%s : %s' % (k, v), extra_specs)
    self.admin_client.unset_share_type_extra_specs(st['ID'], ('foo_key',), microversion=microversion)
    extra_specs = self.admin_client.list_share_type_extra_specs(st['ID'], microversion=microversion)
    self.assertNotIn('foo_key : foo_value', extra_specs)
    self.assertIn('bar_key : bar_value', extra_specs)
    self.assertIn('driver_handles_share_servers : %s' % dhss, extra_specs)