import yaml
from tempest.lib import exceptions
from heatclient.tests.functional.osc.v1 import base
def test_openstack_template_version_list(self):
    ret = self.openstack('orchestration template version list')
    tmpl_types = self.parser.listing(ret)
    self.assertTableStruct(tmpl_types, ['Version', 'Type'])