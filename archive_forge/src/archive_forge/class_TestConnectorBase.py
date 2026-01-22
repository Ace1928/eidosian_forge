import sys
from unittest import mock
import ddt
from glance_store._drivers.cinder import base
from glance_store._drivers.cinder import scaleio
from glance_store.tests import base as test_base
from glance_store._drivers.cinder import store as cinder # noqa
from glance_store._drivers.cinder import nfs # noqa
@ddt.ddt
class TestConnectorBase(test_base.StoreBaseTest):

    @ddt.data(('iscsi', base.BaseBrickConnectorInterface), ('nfs', nfs.NfsBrickConnector), ('scaleio', scaleio.ScaleIOBrickConnector))
    @ddt.unpack
    def test_factory(self, protocol, expected_class):
        connector_class = base.factory(connection_info={'driver_volume_type': protocol})
        self.assertIsInstance(connector_class, expected_class)