from oslo_config import cfg
import testscenarios
from unittest import mock
import oslo_messaging
from oslo_messaging import exceptions
from oslo_messaging import serializer as msg_serializer
from oslo_messaging.tests import utils as test_utils
def test_version_cap(self):
    self.config(rpc_response_timeout=None)
    transport = oslo_messaging.get_rpc_transport(self.conf, url='fake:')
    target = oslo_messaging.Target(version=self.version)
    client = oslo_messaging.get_rpc_client(transport, target, version_cap=self.cap)
    prep_kwargs = {}
    if self.prepare_cap is not _notset:
        prep_kwargs['version_cap'] = self.prepare_cap
    if self.prepare_version is not _notset:
        prep_kwargs['version'] = self.prepare_version
    if prep_kwargs:
        client = client.prepare(**prep_kwargs)
    if self.can_send_version is not _notset:
        can_send = client.can_send_version(version=self.can_send_version)
        call_context_can_send = client.prepare().can_send_version(version=self.can_send_version)
        self.assertEqual(can_send, call_context_can_send)
    else:
        can_send = client.can_send_version()
    self.assertEqual(self.can_send, can_send)