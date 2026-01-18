import itertools
import json
import time
import uuid
from tempest.lib import exceptions
from openstackclient.compute.v2 import server as v2_server
from openstackclient.tests.functional.compute.v2 import common
from openstackclient.tests.functional.volume.v2 import common as volume_common
def test_server_boot_with_bdm_volume_legacy(self):
    """Test server create from image with bdm volume, server delete"""
    self._test_server_boot_with_bdm_volume(use_legacy=True)