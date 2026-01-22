import tempfile
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.functional import base
def _cleanup_stack(self):
    self.user_cloud.delete_stack(self.stack_name, wait=True)
    self.assertIsNone(self.user_cloud.get_stack(self.stack_name))