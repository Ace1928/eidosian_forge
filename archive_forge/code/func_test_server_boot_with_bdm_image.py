import itertools
import json
import time
import uuid
from tempest.lib import exceptions
from openstackclient.compute.v2 import server as v2_server
from openstackclient.tests.functional.compute.v2 import common
from openstackclient.tests.functional.volume.v2 import common as volume_common
def test_server_boot_with_bdm_image(self):
    self._test_server_boot_with_bdm_image(use_legacy=False)