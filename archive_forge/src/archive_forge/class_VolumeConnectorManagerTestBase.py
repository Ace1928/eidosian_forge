import copy
import testtools
from ironicclient import exc
from ironicclient.tests.unit import utils
import ironicclient.v1.port
class VolumeConnectorManagerTestBase(testtools.TestCase):

    def _validate_obj(self, expect, obj):
        self.assertEqual(expect['uuid'], obj.uuid)
        self.assertEqual(expect['type'], obj.type)
        self.assertEqual(expect['connector_id'], obj.connector_id)
        self.assertEqual(expect['node_uuid'], obj.node_uuid)

    def _validate_list(self, expect_request, expect_connectors, actual_connectors):
        self.assertEqual(expect_request, self.api.calls)
        self.assertEqual(len(expect_connectors), len(actual_connectors))
        for expect, obj in zip(expect_connectors, actual_connectors):
            self._validate_obj(expect, obj)