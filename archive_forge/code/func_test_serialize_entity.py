from unittest import mock
from oslo_config import cfg
import oslo_messaging as messaging
from oslo_messaging import conffixture as messaging_conffixture
from oslo_messaging import exceptions as oslomsg_exc
import testtools
from neutron_lib import fixture
from neutron_lib import rpc
from neutron_lib.tests import _base as base
def test_serialize_entity(self):
    self.mock_base.serialize_entity.return_value = 'foo'
    ser_ent = self.ser.serialize_entity('context', 'entity')
    self.mock_base.serialize_entity.assert_called_once_with('context', 'entity')
    self.assertEqual('foo', ser_ent)