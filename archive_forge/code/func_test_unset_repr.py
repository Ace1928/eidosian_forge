from copy import deepcopy
from oslo_utils import uuidutils
from octavia_lib.api.drivers import data_models
from octavia_lib.common import constants
from octavia_lib.tests.unit import base
def test_unset_repr(self):
    self.assertEqual('Unset', repr(data_models.Unset))