from unittest import mock
from heatclient import exc
from heatclient.osc.v1 import template
from heatclient.tests.unit.osc.v1 import fakes
from heatclient.v1 import template_versions
def _stub_versions_list(self, ret_data):
    tv1 = template_versions.TemplateVersion(None, ret_data[0])
    tv2 = template_versions.TemplateVersion(None, ret_data[1])
    self.template_versions.list.return_value = [tv1, tv2]
    self.cmd = template.VersionList(self.app, None)