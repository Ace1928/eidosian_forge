import tempfile
from tensorflow.core.protobuf import data_service_pb2
from tensorflow.core.protobuf import service_config_pb2
from tensorflow.python.data.experimental.kernel_tests.service import test_base as data_service_test_base
from tensorflow.python.data.experimental.service import server_lib
from tensorflow.python.distribute import multi_process_lib
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest
def _start_dispatcher(self, worker_addresses, port=0):
    if port == 0:
        port = test_util.pick_unused_port()
    self._dispatcher = server_lib.DispatchServer(service_config_pb2.DispatcherConfig(port=port, protocol='grpc', work_dir=self._work_dir, fault_tolerant_mode=True, worker_addresses=worker_addresses, deployment_mode=self._deployment_mode), start=True)