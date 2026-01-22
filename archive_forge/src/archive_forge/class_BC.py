from typing import Type, Union
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.bc.bc_catalog import BCCatalog
from ray.rllib.algorithms.marwil.marwil import MARWIL, MARWILConfig
from ray.rllib.core.learner import Learner
from ray.rllib.core.learner.learner_group_config import ModuleSpec
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.utils.annotations import override
from ray.rllib.utils.metrics import (
from ray.rllib.utils.typing import ResultDict
class BC(MARWIL):
    """Behavioral Cloning (derived from MARWIL).

    Simply uses MARWIL with beta force-set to 0.0.
    """

    @classmethod
    @override(MARWIL)
    def get_default_config(cls) -> AlgorithmConfig:
        return BCConfig()

    @override(MARWIL)
    def training_step(self) -> ResultDict:
        if not self.config._enable_new_api_stack:
            return super().training_step()
        else:
            with self._timers[SAMPLE_TIMER]:
                if self.config.count_steps_by == 'agent_steps':
                    train_batch = synchronous_parallel_sample(worker_set=self.workers, max_agent_steps=self.config.train_batch_size)
                else:
                    train_batch = synchronous_parallel_sample(worker_set=self.workers, max_env_steps=self.config.train_batch_size)
                train_batch = train_batch.as_multi_agent()
                self._counters[NUM_AGENT_STEPS_SAMPLED] += train_batch.agent_steps()
                self._counters[NUM_ENV_STEPS_SAMPLED] += train_batch.env_steps()
            is_module_trainable = self.workers.local_worker().is_policy_to_train
            self.learner_group.set_is_module_trainable(is_module_trainable)
            train_results = self.learner_group.update(train_batch)
            policies_to_update = set(train_results.keys()) - {ALL_MODULES}
            global_vars = {'timestep': self._counters[NUM_AGENT_STEPS_SAMPLED]}
            with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
                if self.workers.num_remote_workers() > 0:
                    self.workers.sync_weights(from_worker_or_learner_group=self.learner_group, policies=policies_to_update, global_vars=global_vars)
                else:
                    self.workers.local_worker().set_weights(self.learner_group.get_weights())
            return train_results