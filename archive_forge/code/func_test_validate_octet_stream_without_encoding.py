from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.resources.openstack.barbican import secret
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_validate_octet_stream_without_encoding(self):
    props = {'name': 'secret', 'payload': 'foobar', 'payload_content_type': 'application/octet-stream'}
    defn = rsrc_defn.ResourceDefinition('secret', 'OS::Barbican::Secret', props)
    res = self._create_resource(defn.name, defn, self.stack)
    msg = "Property unspecified. For 'application/octet-stream' value of 'payload_content_type' property, 'payload_content_encoding' property must be specified."
    self.assertRaisesRegex(exception.StackValidationFailed, msg, res.validate)