import logging
import threading
import time
from typing import Union, Optional
from enum import Enum
import ray.cloudpickle as pickle
from ray.rllib.env.external_env import ExternalEnv
from ray.rllib.env.external_multi_agent_env import ExternalMultiAgentEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.utils.typing import (
class _LocalInferenceThread(threading.Thread):
    """Thread that handles experience generation (worker.sample() loop)."""

    def __init__(self, rollout_worker, send_fn):
        super().__init__()
        self.daemon = True
        self.rollout_worker = rollout_worker
        self.send_fn = send_fn

    def run(self):
        try:
            while True:
                logger.info('Generating new batch of experiences.')
                samples = self.rollout_worker.sample()
                metrics = self.rollout_worker.get_metrics()
                if isinstance(samples, MultiAgentBatch):
                    logger.info('Sending batch of {} env steps ({} agent steps) to server.'.format(samples.env_steps(), samples.agent_steps()))
                else:
                    logger.info('Sending batch of {} steps back to server.'.format(samples.count))
                self.send_fn({'command': Commands.REPORT_SAMPLES, 'samples': samples, 'metrics': metrics})
        except Exception as e:
            logger.error('Error: inference worker thread died!', e)