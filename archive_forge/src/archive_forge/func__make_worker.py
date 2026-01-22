import tempfile
from tensorflow.core.protobuf import service_config_pb2
from tensorflow.python.data.experimental.ops import data_service_ops
from tensorflow.python.data.experimental.service import server_lib
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest
def _make_worker(dispatcher_address, protocol, data_transfer_protocol, shutdown_quiet_period_ms=0, port=0, worker_tags=None, cross_trainer_cache_size_bytes=None, snapshot_max_chunk_size_bytes=TEST_SNAPSHOT_MAX_CHUNK_SIZE_BYTES):
    """Creates a worker server."""
    defaults = server_lib.WorkerConfig(dispatcher_address=dispatcher_address)
    config_proto = service_config_pb2.WorkerConfig(dispatcher_address=dispatcher_address, worker_address=defaults.worker_address, port=port, protocol=protocol, worker_tags=worker_tags, heartbeat_interval_ms=TEST_HEARTBEAT_INTERVAL_MS, dispatcher_timeout_ms=TEST_DISPATCHER_TIMEOUT_MS, data_transfer_protocol=data_transfer_protocol, data_transfer_address=defaults.worker_address, shutdown_quiet_period_ms=shutdown_quiet_period_ms, cross_trainer_cache_size_bytes=cross_trainer_cache_size_bytes, snapshot_max_chunk_size_bytes=snapshot_max_chunk_size_bytes)
    return server_lib.WorkerServer(config_proto, start=False)