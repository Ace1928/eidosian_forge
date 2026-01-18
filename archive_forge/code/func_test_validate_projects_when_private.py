import collections
from unittest import mock
from heat.common import exception
from heat.engine.clients.os import cinder as c_plugin
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_validate_projects_when_private(self):
    tmpl = self.stack.t.t
    props = tmpl['resources']['my_volume_type']['properties'].copy()
    props['is_public'] = False
    props['projects'] = ['id1']
    self.my_volume_type.t = self.my_volume_type.t.freeze(properties=props)
    self.my_volume_type.reparse()
    self.cinderclient.volume_api_version = 3
    self.stub_KeystoneProjectConstraint()
    self.assertIsNone(self.my_volume_type.validate())