import datetime
from unittest import mock
from oslo_config import cfg
from oslo_utils import timeutils
from heat.common import context
from heat.common import service_utils
from heat.engine import service
from heat.engine import worker
from heat.objects import service as service_objects
from heat.rpc import worker_api
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import utils
def _test_engine_service_start(self, thread_group_class, worker_service_class, engine_listener_class, thread_group_manager_class, sample_uuid_method, rpc_client_class, target_class, rpc_server_method):
    self.patchobject(self.eng, 'service_manage_cleanup')
    self.patchobject(self.eng, 'reset_stack_status')
    self.eng.start()
    sample_uuid_method.assert_called_once_with()
    sampe_uuid = sample_uuid_method.return_value
    self.assertEqual(sampe_uuid, self.eng.engine_id, 'Failed to generated engine_id')
    thread_group_manager_class.assert_called_once_with()
    thread_group_manager = thread_group_manager_class.return_value
    self.assertEqual(thread_group_manager, self.eng.thread_group_mgr, 'Failed to create Thread Group Manager')
    engine_listener_class.assert_called_once_with(self.eng.host, self.eng.engine_id, self.eng.thread_group_mgr)
    engine_lister = engine_listener_class.return_value
    engine_lister.start.assert_called_once_with()
    if cfg.CONF.convergence_engine:
        worker_service_class.assert_called_once_with(host=self.eng.host, topic=worker_api.TOPIC, engine_id=self.eng.engine_id, thread_group_mgr=self.eng.thread_group_mgr)
        worker_service = worker_service_class.return_value
        worker_service.start.assert_called_once_with()
    target_class.assert_called_once_with(version=service.EngineService.RPC_API_VERSION, server=self.eng.host, topic=self.eng.topic)
    target = target_class.return_value
    rpc_server_method.assert_called_once_with(target, self.eng)
    rpc_server = rpc_server_method.return_value
    self.assertEqual(rpc_server, self.eng._rpc_server, 'Failed to create RPC server')
    rpc_server.start.assert_called_once_with()
    rpc_client = rpc_client_class.return_value
    rpc_client_class.assert_called_once_with(version=service.EngineService.RPC_API_VERSION)
    self.assertEqual(rpc_client, self.eng._client, 'Failed to create RPC client')
    thread_group_class.assert_called_once_with()
    manage_thread_group = thread_group_class.return_value
    manage_thread_group.add_timer.assert_called_once_with(cfg.CONF.periodic_interval, self.eng.service_manage_report)