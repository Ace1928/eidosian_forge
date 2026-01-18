import testscenarios
import time
import oslo_messaging
from oslo_messaging import rpc
from oslo_messaging import serializer as msg_serializer
from oslo_messaging.tests import utils as test_utils
from unittest import mock
def test_serializer(self):
    endpoint = _FakeEndpoint()
    serializer = msg_serializer.NoOpSerializer()
    dispatcher = oslo_messaging.RPCDispatcher([endpoint], serializer)
    endpoint.foo = mock.Mock()
    args = dict([(k, 'd' + v) for k, v in self.args.items()])
    endpoint.foo.return_value = self.retval
    serializer.serialize_entity = mock.Mock()
    serializer.deserialize_entity = mock.Mock()
    serializer.deserialize_context = mock.Mock()
    serializer.deserialize_context.return_value = self.dctxt
    expected_side_effect = ['d' + arg for arg in self.args]
    serializer.deserialize_entity.side_effect = expected_side_effect
    serializer.serialize_entity.return_value = None
    if self.retval:
        serializer.serialize_entity.return_value = 's' + self.retval
    incoming = mock.Mock()
    incoming.ctxt = self.ctxt
    incoming.message = dict(method='foo', args=self.args)
    incoming.client_timeout = 0
    retval = dispatcher.dispatch(incoming)
    if self.retval is not None:
        self.assertEqual('s' + self.retval, retval)
    endpoint.foo.assert_called_once_with(self.dctxt, **args)
    serializer.deserialize_context.assert_called_once_with(self.ctxt)
    expected_calls = [mock.call(self.dctxt, arg) for arg in self.args]
    self.assertEqual(expected_calls, serializer.deserialize_entity.mock_calls)
    serializer.serialize_entity.assert_called_once_with(self.dctxt, self.retval)