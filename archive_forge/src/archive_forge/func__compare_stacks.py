import tempfile
import testtools
from openstack import exceptions
from openstack.orchestration.v1 import stack
from openstack.tests import fakes
from openstack.tests.unit import base
def _compare_stacks(self, exp, real):
    self.assertDictEqual(stack.Stack(**exp).to_dict(computed=False), real.to_dict(computed=False))