import testtools
from osc_lib import exceptions
from osc_lib.tests import utils
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
def _mock_fwaas(*args, **kwargs):
    return {'id': args[0]}