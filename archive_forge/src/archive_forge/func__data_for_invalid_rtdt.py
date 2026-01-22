from neutron_lib.api.definitions import bgpvpn
from neutron_lib.api import validators
from neutron_lib.tests.unit.api.definitions import base
def _data_for_invalid_rtdt(self):
    values = [[':1'], ['1:'], ['42'], ['65536:123456'], ['123.456.789.123:65535'], ['4294967296:65535'], ['1.1.1.1:655351'], ['4294967295:65536'], ['']]
    for value in values:
        yield value