from unittest import mock
from heat.rpc import worker_api as rpc_api
from heat.rpc import worker_client as rpc_client
from heat.tests import common
class WorkerClientTest(common.HeatTestCase):

    def setUp(self):
        super(WorkerClientTest, self).setUp()
        self.fake_engine_id = 'fake-engine-id'

    def test_make_msg(self):
        method = 'sample_method'
        kwargs = {'a': '1', 'b': '2'}
        result = (method, kwargs)
        self.assertEqual(result, rpc_client.WorkerClient.make_msg(method, **kwargs))

    @mock.patch('heat.common.messaging.get_rpc_client', return_value=mock.Mock())
    def test_cast(self, rpc_client_method):
        mock_rpc_client = rpc_client_method.return_value
        worker_client = rpc_client.WorkerClient()
        rpc_client_method.assert_called_once_with(version=rpc_client.WorkerClient.BASE_RPC_API_VERSION, topic=rpc_api.TOPIC)
        self.assertEqual(mock_rpc_client, worker_client._client, 'Failed to create RPC client')
        mock_cnxt = mock.Mock()
        method = 'sample_method'
        kwargs = {'a': '1', 'b': '2'}
        msg = (method, kwargs)
        return_value = worker_client.cast(mock_cnxt, msg)
        self.assertIsNone(return_value)
        mock_rpc_client.cast.assert_called_with(mock_cnxt, method, **kwargs)
        version = '1.3'
        return_value = worker_client.cast(mock_cnxt, msg, version)
        self.assertIsNone(return_value)
        mock_rpc_client.prepare.assert_called_once_with(version=version)
        mock_rpc_client.cast.assert_called_once_with(mock_cnxt, method, **kwargs)

    def test_cancel_check_resource(self):
        mock_stack_id = 'dummy-stack-id'
        mock_cnxt = mock.Mock()
        method = 'cancel_check_resource'
        kwargs = {'stack_id': mock_stack_id}
        mock_rpc_client = mock.MagicMock()
        mock_cast = mock.MagicMock()
        with mock.patch('heat.common.messaging.get_rpc_client') as mock_grc:
            mock_grc.return_value = mock_rpc_client
            mock_rpc_client.prepare.return_value = mock_cast
            wc = rpc_client.WorkerClient()
            ret_val = wc.cancel_check_resource(mock_cnxt, mock_stack_id, self.fake_engine_id)
            mock_grc.assert_called_with(version=wc.BASE_RPC_API_VERSION, topic=rpc_api.TOPIC, server=self.fake_engine_id)
            self.assertIsNone(ret_val)
            mock_rpc_client.prepare.assert_called_with(version='1.3')
            mock_cast.cast.assert_called_with(mock_cnxt, method, **kwargs)