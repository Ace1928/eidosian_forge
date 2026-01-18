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
Initializes a MultiGPULearnerThread instance.

        Args:
            local_worker: Local RolloutWorker holding
                policies this thread will call `load_batch_into_buffer` and
                `learn_on_loaded_batch` on.
            num_gpus: Number of GPUs to use for data-parallel SGD.
            train_batch_size: Size of batches (minibatches if
                `num_sgd_iter` > 1) to learn on.
            num_multi_gpu_tower_stacks: Number of buffers to parallelly
                load data into on one device. Each buffer is of size of
                `train_batch_size` and hence increases GPU memory usage
                accordingly.
            num_sgd_iter: Number of passes to learn on per train batch
                (minibatch if `num_sgd_iter` > 1).
            learner_queue_size: Max size of queue of inbound
                train batches to this thread.
            num_data_load_threads: Number of threads to use to load
                data into GPU memory in parallel.
        