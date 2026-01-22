from collections import deque
from http.server import HTTPServer, SimpleHTTPRequestHandler
import logging
import queue
from socketserver import ThreadingMixIn
import threading
import time
import traceback
from typing import List
import ray.cloudpickle as pickle
from ray.rllib.env.policy_client import (
from ray.rllib.offline.input_reader import InputReader
from ray.rllib.offline.io_context import IOContext
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.evaluation.metrics import RolloutMetrics
from ray.rllib.evaluation.sampler import SamplerInput
from ray.rllib.utils.typing import SampleBatchType
@PublicAPI
class PolicyServerInput(ThreadingMixIn, HTTPServer, InputReader):
    """REST policy server that acts as an offline data source.

    This launches a multi-threaded server that listens on the specified host
    and port to serve policy requests and forward experiences to RLlib. For
    high performance experience collection, it implements InputReader.

    For an example, run `examples/serving/cartpole_server.py` along
    with `examples/serving/cartpole_client.py --inference-mode=local|remote`.

    WARNING: This class is not meant to be publicly exposed. Anyone that can
    communicate with this server can execute arbitary code on the machine. Use
    this with caution, in isolated environments, and at your own risk.

    .. testcode::
        :skipif: True

        import gymnasium as gym
        from ray.rllib.algorithms.ppo import PPOConfig
        from ray.rllib.env.policy_client import PolicyClient
        from ray.rllib.env.policy_server_input import PolicyServerInput
        addr, port = ...
        config = (
            PPOConfig()
            .environment("CartPole-v1")
            .offline_data(
                input_=lambda ioctx: PolicyServerInput(ioctx, addr, port)
            )
            # Run just 1 server (in the Algorithm's WorkerSet).
            .rollouts(num_rollout_workers=0)
        )
        algo = config.build()
        while True:
            algo.train()
        client = PolicyClient(
            "localhost:9900", inference_mode="local")
        eps_id = client.start_episode()
        env = gym.make("CartPole-v1")
        obs, info = env.reset()
        action = client.get_action(eps_id, obs)
        _, reward, _, _, _ = env.step(action)
        client.log_returns(eps_id, reward)
        client.log_returns(eps_id, reward)
        algo.stop()
    """

    @PublicAPI
    def __init__(self, ioctx: IOContext, address: str, port: int, idle_timeout: float=3.0, max_sample_queue_size: int=20):
        """Create a PolicyServerInput.

        This class implements rllib.offline.InputReader, and can be used with
        any Algorithm by configuring

        [AlgorithmConfig object]
        .rollouts(num_rollout_workers=0)
        .offline_data(input_=lambda ioctx: PolicyServerInput(ioctx, addr, port))

        Note that by setting num_rollout_workers: 0, the algorithm will only create one
        rollout worker / PolicyServerInput. Clients can connect to the launched
        server using rllib.env.PolicyClient. You can increase the number of available
        connections (ports) by setting num_rollout_workers to a larger number. The ports
        used will then be `port` + the worker's index.

        Args:
            ioctx: IOContext provided by RLlib.
            address: Server addr (e.g., "localhost").
            port: Server port (e.g., 9900).
            max_queue_size: The maximum size for the sample queue. Once full, will
                purge (throw away) 50% of all samples, oldest first, and continue.
        """
        self.rollout_worker = ioctx.worker
        self.samples_queue = deque(maxlen=max_sample_queue_size)
        self.metrics_queue = queue.Queue()
        self.idle_timeout = idle_timeout
        if self.rollout_worker.sampler is not None:

            def get_metrics():
                completed = []
                while True:
                    try:
                        completed.append(self.metrics_queue.get_nowait())
                    except queue.Empty:
                        break
                return completed
            self.rollout_worker.sampler.get_metrics = get_metrics
        else:

            class MetricsDummySampler(SamplerInput):
                """This sampler only maintains a queue to get metrics from."""

                def __init__(self, metrics_queue):
                    """Initializes a MetricsDummySampler instance.

                    Args:
                        metrics_queue: A queue of metrics
                    """
                    self.metrics_queue = metrics_queue

                def get_data(self) -> SampleBatchType:
                    raise NotImplementedError

                def get_extra_batches(self) -> List[SampleBatchType]:
                    raise NotImplementedError

                def get_metrics(self) -> List[RolloutMetrics]:
                    """Returns metrics computed on a policy client rollout worker."""
                    completed = []
                    while True:
                        try:
                            completed.append(self.metrics_queue.get_nowait())
                        except queue.Empty:
                            break
                    return completed
            self.rollout_worker.sampler = MetricsDummySampler(self.metrics_queue)
        handler = _make_handler(self.rollout_worker, self.samples_queue, self.metrics_queue)
        try:
            import time
            time.sleep(1)
            HTTPServer.__init__(self, (address, port), handler)
        except OSError:
            print(f'Creating a PolicyServer on {address}:{port} failed!')
            import time
            time.sleep(1)
            raise
        logger.info(f'Starting connector server at {self.server_name}:{self.server_port}')
        serving_thread = threading.Thread(name='server', target=self.serve_forever)
        serving_thread.daemon = True
        serving_thread.start()
        heart_beat_thread = threading.Thread(name='heart-beat', target=self._put_empty_sample_batch_every_n_sec)
        heart_beat_thread.daemon = True
        heart_beat_thread.start()

    @override(InputReader)
    def next(self):
        while len(self.samples_queue) == 0:
            time.sleep(0.1)
        return self.samples_queue.pop()

    def _put_empty_sample_batch_every_n_sec(self):
        while True:
            time.sleep(self.idle_timeout)
            self.samples_queue.append(SampleBatch())