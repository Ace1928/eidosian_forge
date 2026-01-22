import copy
import testtools
from openstack import exceptions
from openstack.network.v2 import port as _port
from openstack.network.v2 import router as _router
from openstack.tests.unit import base
def _compare_routers(self, exp, real):
    self.assertDictEqual(_router.Router(**exp).to_dict(computed=False), real.to_dict(computed=False))