import logging
import queue
import threading
from ray.util.timer import _Timer
from ray.rllib.execution.learner_thread import LearnerThread
from ray.rllib.execution.minibatch_buffer import MinibatchBuffer
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import deprecation_warning
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.metrics.learner_info import LearnerInfoBuilder
from ray.rllib.evaluation.rollout_worker import RolloutWorker
class _MultiGPULoaderThread(threading.Thread):

    def __init__(self, multi_gpu_learner_thread: MultiGPULearnerThread, share_stats: bool):
        threading.Thread.__init__(self)
        self.multi_gpu_learner_thread = multi_gpu_learner_thread
        self.daemon = True
        if share_stats:
            self.queue_timer = multi_gpu_learner_thread.queue_timer
            self.load_timer = multi_gpu_learner_thread.load_timer
        else:
            self.queue_timer = _Timer()
            self.load_timer = _Timer()

    def run(self) -> None:
        while True:
            self._step()

    def _step(self) -> None:
        s = self.multi_gpu_learner_thread
        policy_map = s.policy_map
        with self.queue_timer:
            batch = s.inqueue.get()
        buffer_idx = s.idle_tower_stacks.get()
        with self.load_timer:
            for pid in policy_map.keys():
                if s.local_worker.is_policy_to_train is not None and (not s.local_worker.is_policy_to_train(pid, batch)):
                    continue
                policy = policy_map[pid]
                if isinstance(batch, SampleBatch):
                    policy.load_batch_into_buffer(batch=batch, buffer_index=buffer_idx)
                elif pid in batch.policy_batches:
                    policy.load_batch_into_buffer(batch=batch.policy_batches[pid], buffer_index=buffer_idx)
        s.ready_tower_stacks.put(buffer_idx)