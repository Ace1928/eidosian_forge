from neutron_lib.api.definitions import bgpvpn
from neutron_lib.api import validators
from neutron_lib.tests.unit.api.definitions import base
class BgpvpnDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = bgpvpn
    extension_resources = (bgpvpn.COLLECTION_NAME,)
    extension_attributes = ('type', 'route_targets', 'import_targets', 'export_targets', 'route_distinguishers', 'networks', 'routers', 'router_id', 'network_id')
    extension_subresources = ('network_associations', 'router_associations')

    def _data_for_invalid_rtdt(self):
        values = [[':1'], ['1:'], ['42'], ['65536:123456'], ['123.456.789.123:65535'], ['4294967296:65535'], ['1.1.1.1:655351'], ['4294967295:65536'], ['']]
        for value in values:
            yield value

    def _data_for_valid_rtdt(self):
        values = [['1:1'], ['1:4294967295'], ['65535:0'], ['65535:4294967295'], ['1.1.1.1:1'], ['1.1.1.1:65535'], ['4294967295:0'], ['65536:65535'], ['4294967295:65535']]
        for value in values:
            yield value

    def test_valid_rtrd(self):
        for rtrd in self._data_for_valid_rtdt():
            msg = validators.validate_list_of_regex_or_none(rtrd, bgpvpn.RTRD_REGEX)
            self.assertIsNone(msg)

    def test_invalid_rtrd(self):
        for rtrd in self._data_for_invalid_rtdt():
            msg = validators.validate_list_of_regex_or_none(rtrd, bgpvpn.RTRD_REGEX)
            self.assertIsNotNone(msg)