import copy
import tempfile
import time
from unittest import mock
import testtools
from testtools.matchers import HasLength
from ironicclient.common import utils as common_utils
from ironicclient import exc
from ironicclient.tests.unit import utils
from ironicclient.v1 import node
from ironicclient.v1 import volume_connector
from ironicclient.v1 import volume_target
def _validate_node_volume_connector_list(self, expect, volume_connectors):
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(1, len(volume_connectors))
    self.assertIsInstance(volume_connectors[0], volume_connector.VolumeConnector)
    self.assertEqual(CONNECTOR['uuid'], volume_connectors[0].uuid)
    self.assertEqual(CONNECTOR['type'], volume_connectors[0].type)
    self.assertEqual(CONNECTOR['connector_id'], volume_connectors[0].connector_id)