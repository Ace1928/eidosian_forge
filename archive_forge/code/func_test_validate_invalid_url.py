from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import sahara
from heat.engine.resources.openstack.sahara import job_binary
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_validate_invalid_url(self):
    props = self.stack.t.t['resources']['job-binary']['properties'].copy()
    props['url'] = 'internal-db://38273f82'
    self.rsrc_defn = self.rsrc_defn.freeze(properties=props)
    jb = job_binary.JobBinary('job-binary', self.rsrc_defn, self.stack)
    ex = self.assertRaises(exception.StackValidationFailed, jb.validate)
    error_msg = 'resources.job-binary.properties: internal-db://38273f82 is not a valid job location.'
    self.assertEqual(error_msg, str(ex))