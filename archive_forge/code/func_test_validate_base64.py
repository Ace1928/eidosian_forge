from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.resources.openstack.barbican import secret
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_validate_base64(self):
    props = {'name': 'secret', 'payload': 'foobar', 'payload_content_type': 'application/octet-stream', 'payload_content_encoding': 'base64'}
    defn = rsrc_defn.ResourceDefinition('secret', 'OS::Barbican::Secret', props)
    res = self._create_resource(defn.name, defn, self.stack)
    msg = "Invalid payload for specified 'base64' value of 'payload_content_encoding' property."
    self.assertRaisesRegex(exception.StackValidationFailed, msg, res.validate)