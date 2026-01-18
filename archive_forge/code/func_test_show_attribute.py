from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.resources.openstack.sahara import data_source
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_show_attribute(self):
    ds = self._create_resource('data-source', self.rsrc_defn, self.stack)
    value = mock.MagicMock()
    value.to_dict.return_value = {'ds': 'info'}
    self.client.data_sources.get.return_value = value
    self.assertEqual({'ds': 'info'}, ds.FnGetAtt('show'))