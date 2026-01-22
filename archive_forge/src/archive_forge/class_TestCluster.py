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
class TestCluster:
    """Test tf.data service cluster."""

    def __init__(self, num_workers, dispatcher_port=0, work_dir=TMP_WORK_DIR, fault_tolerant_mode=True, job_gc_check_interval_ms=TEST_JOB_GC_CHECK_INTERNAL_MS, job_gc_timeout_ms=None, worker_timeout_ms=TEST_WORKER_TIMEOUT_MS, worker_shutdown_quiet_period_ms=0, snapshot_max_chunk_size_bytes=TEST_SNAPSHOT_MAX_CHUNK_SIZE_BYTES, worker_max_concurrent_snapshots=0, start=True, protocol=PROTOCOL, data_transfer_protocol=None):
        """Creates a tf.data service test cluster.

    Args:
      num_workers: The number of workers to initially add to the cluster.
      dispatcher_port: The port to use for the dispatcher.
      work_dir: The work directory to use for the dispatcher. If set to
        `TMP_WORK_DIR`, the cluster will create a new temporary directory to use
        as the work directory. If set to `NO_WORK_DIR`, no work directory will
        be used.
      fault_tolerant_mode: Whether the dispatcher should write its state to a
        journal so that it can recover from restarts.
      job_gc_check_interval_ms: How often the dispatcher should scan through to
        delete old and unused jobs, in milliseconds.
      job_gc_timeout_ms: How long a job needs to be unused before it becomes a
        candidate for garbage collection, in milliseconds.
      worker_timeout_ms: How long to wait for a worker to heartbeat before
        considering it missing, in milliseconds.
      worker_shutdown_quiet_period_ms: When shutting down a worker, how long to
        wait for the gRPC server to process the final requests.
      snapshot_max_chunk_size_bytes: The maximum size of a distributed snapshot
        chunk file.
      worker_max_concurrent_snapshots: The maximum number of snapshots a worker
        can concurrently process.
      start: Whether to immediately start the servers in the cluster. If
        `False`, the servers can be started later by calling
        `start_dispatcher()` and `start_workers()`.
      protocol: The protocol to use for communicating with the tf.data service,
        e.g. "grpc".
      data_transfer_protocol: (Optional.) The protocol to use for transferring
        data with the tf.data service.
    """
        if work_dir == TMP_WORK_DIR:
            work_dir = tempfile.mkdtemp(dir=googletest.GetTempDir())
        self._worker_shutdown_quiet_period_ms = worker_shutdown_quiet_period_ms
        self._snapshot_max_chunk_size_bytes = snapshot_max_chunk_size_bytes
        self._protocol = protocol
        self._data_transfer_protocol = data_transfer_protocol
        self._job_gc_check_interval_ms = job_gc_check_interval_ms
        self._job_gc_timeout_ms = job_gc_timeout_ms
        self._worker_timeout_ms = worker_timeout_ms
        self._worker_max_concurrent_snapshots = worker_max_concurrent_snapshots
        self.dispatcher = server_lib.DispatchServer(server_lib.DispatcherConfig(port=dispatcher_port, work_dir=work_dir, protocol=protocol, fault_tolerant_mode=fault_tolerant_mode, job_gc_check_interval_ms=job_gc_check_interval_ms, job_gc_timeout_ms=job_gc_timeout_ms, worker_timeout_ms=worker_timeout_ms, worker_max_concurrent_snapshots=worker_max_concurrent_snapshots), start=start)
        self.workers = []
        for _ in range(num_workers):
            self.add_worker(start=start)

    def dispatcher_address(self):
        return self.dispatcher.target.split('://')[1]

    def add_worker(self, start=True):
        worker = TestWorker(self.dispatcher_address(), self._worker_shutdown_quiet_period_ms, self._protocol, self._data_transfer_protocol, snapshot_max_chunk_size_bytes=self._snapshot_max_chunk_size_bytes)
        if start:
            worker.start()
        self.workers.append(worker)

    def start_dispatcher(self):
        self.dispatcher.start()

    def start_workers(self):
        for worker in self.workers:
            worker.start()

    def stop_dispatcher(self):
        self.dispatcher._stop()

    def restart_worker(self, index):
        self.workers[index].restart()

    def stop_worker(self, index):
        self.workers[index].stop()

    def stop_workers(self):
        for worker in self.workers:
            worker.stop()

    def restart_dispatcher(self):
        """Stops `dispatcher` and creates a new dispatcher with the same port.

    Restarting is supported only when the dispatcher is configured with
    `fault_tolerant_mode=True`.
    """
        if not self.dispatcher._config.fault_tolerant_mode:
            raise ValueError('Trying to restart the dispatcher without fault-tolerance.')
        port = int(self.dispatcher_address().split(':')[1])
        self.dispatcher._stop()
        self.dispatcher = server_lib.DispatchServer(server_lib.DispatcherConfig(port=port, work_dir=self.dispatcher._config.work_dir, protocol=self._protocol, fault_tolerant_mode=self.dispatcher._config.fault_tolerant_mode, job_gc_check_interval_ms=self._job_gc_check_interval_ms, job_gc_timeout_ms=self._job_gc_timeout_ms, worker_timeout_ms=self._worker_timeout_ms, worker_max_concurrent_snapshots=self._worker_max_concurrent_snapshots))

    def num_registered_workers(self):
        return self.dispatcher._num_workers()

    def num_tasks_on_workers(self):
        return sum((worker.num_tasks() for worker in self.workers))

    def snapshot_streams(self, path):
        return self.dispatcher._snapshot_streams(path)

    def __del__(self):
        self.workers.clear()
        del self.dispatcher