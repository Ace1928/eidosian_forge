import time
from oslo_utils import timeutils
from oslo_utils import uuidutils
from tempest.lib import exceptions
from novaclient.tests.functional import base
def test_show_action_with_not_existing_instance(self):
    name_or_uuid = uuidutils.generate_uuid()
    request_id = uuidutils.generate_uuid()
    self._test_cmd_with_not_existing_instance('instance-action', '%s %s' % (name_or_uuid, request_id))