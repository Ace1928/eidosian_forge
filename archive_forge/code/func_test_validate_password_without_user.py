from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.resources.openstack.sahara import data_source
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_validate_password_without_user(self):
    props = self.stack.t.t['resources']['data-source']['properties'].copy()
    del props['credentials']['user']
    self.rsrc_defn = self.rsrc_defn.freeze(properties=props)
    ds = data_source.DataSource('data-source', self.rsrc_defn, self.stack)
    ex = self.assertRaises(exception.StackValidationFailed, ds.validate)
    error_msg = 'Property error: resources.data-source.properties.credentials: Property user not assigned'
    self.assertEqual(error_msg, str(ex))