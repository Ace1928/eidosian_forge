from collections import defaultdict, deque
from functools import partial
import pathlib
from typing import (
import uuid
import ray
from ray.rllib.core.learner.reduce_result_dict_fn import _reduce_mean_results
from ray.rllib.core.rl_module.rl_module import (
from ray.rllib.core.learner.learner import LearnerSpec
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.utils.actor_manager import FaultTolerantActorManager
from ray.rllib.utils.minibatch_utils import ShardBatchIterator
from ray.rllib.utils.typing import ResultDict
from ray.rllib.utils.numpy import convert_to_numpy
from ray.train._internal.backend_executor import BackendExecutor
from ray.tune.utils.file_transfer import sync_dir_between_nodes
from ray.util.annotations import PublicAPI
def _get_async_results(self, results):
    """Get results from the worker manager and group them by tag.

        Returns:
            A list of lists of results, where each inner list contains all results
            for same tags.

        """
    unprocessed_results = defaultdict(list)
    for result in results:
        result_or_error = result.get()
        if result.ok:
            assert result.tag, 'Cannot call _get_async_results on untagged async requests.'
            unprocessed_results[result.tag].append(result_or_error)
        else:
            raise result_or_error
    for tag in unprocessed_results.keys():
        self._inflight_request_tags.remove(tag)
    return list(unprocessed_results.values())