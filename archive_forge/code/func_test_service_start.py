from unittest import mock
from heat.db import api as db_api
from heat.engine import check_resource
from heat.engine import stack as parser
from heat.engine import template as templatem
from heat.engine import worker
from heat.objects import stack as stack_objects
from heat.rpc import worker_client as wc
from heat.tests import common
from heat.tests import utils
@mock.patch('heat.common.messaging.get_rpc_server', return_value=mock.Mock())
@mock.patch('oslo_messaging.Target', return_value=mock.Mock())
@mock.patch('heat.rpc.worker_client.WorkerClient', return_value=mock.Mock())
def test_service_start(self, rpc_client_class, target_class, rpc_server_method):
    self.worker = worker.WorkerService('host-1', 'topic-1', 'engine_id', mock.Mock())
    self.worker.start()
    target_class.assert_called_once_with(version=worker.WorkerService.RPC_API_VERSION, server=self.worker.engine_id, topic=self.worker.topic)
    target = target_class.return_value
    rpc_server_method.assert_called_once_with(target, self.worker)
    rpc_server = rpc_server_method.return_value
    self.assertEqual(rpc_server, self.worker._rpc_server, 'Failed to create RPC server')
    rpc_server.start.assert_called_once_with()
    rpc_client = rpc_client_class.return_value
    rpc_client_class.assert_called_once_with()
    self.assertEqual(rpc_client, self.worker._rpc_client, 'Failed to create RPC client')