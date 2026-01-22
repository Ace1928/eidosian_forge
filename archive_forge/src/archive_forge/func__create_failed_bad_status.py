from unittest import mock
import uuid
from oslo_config import cfg
from troveclient import exceptions as troveexc
from troveclient.v1 import users
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.clients.os import trove
from heat.engine import resource
from heat.engine.resources.openstack.trove import instance as dbinstance
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import template as tmpl
from heat.tests import common
from heat.tests import utils
def _create_failed_bad_status(self, status, error_message):
    t = template_format.parse(db_template)
    bad_instance = mock.Mock()
    bad_instance.status = status
    self.client.instances.get.return_value = bad_instance
    instance = self._setup_test_instance('test_bad_statuses', t)
    ex = self.assertRaises(exception.ResourceInError, instance.check_create_complete, self.fake_instance.id)
    self.assertIn(error_message, str(ex))