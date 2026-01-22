import math
from ray.rllib.policy.sample_batch import MultiAgentBatch, concat_samples
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.policy.sample_batch import SampleBatch
class MiniBatchDummyIterator(MiniBatchIteratorBase):

    def __init__(self, batch: MultiAgentBatch, minibatch_size: int, num_iters: int=1):
        super().__init__(batch, minibatch_size, num_iters)
        self._batch = batch

    def __iter__(self):
        yield self._batch