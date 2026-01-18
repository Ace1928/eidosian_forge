from unittest import mock
from cinderclient.v3 import volume_snapshots
from cinderclient.v3 import volumes
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit import utils as test_utils
from openstackclient.volume import client  # noqa
def test_not_find(self):
    self.assertRaises(exceptions.CommandError, utils.find_resource, self.manager, 'GeorgeMartin')