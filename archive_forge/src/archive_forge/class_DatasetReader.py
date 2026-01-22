import logging
import math
from pathlib import Path
import re
import numpy as np
from typing import List, Tuple, TYPE_CHECKING, Optional
import zipfile
import ray.data
from ray.rllib.offline.input_reader import InputReader
from ray.rllib.offline.io_context import IOContext
from ray.rllib.offline.json_reader import from_json_data, postprocess_actions
from ray.rllib.policy.sample_batch import concat_samples, SampleBatch, DEFAULT_POLICY_ID
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.typing import SampleBatchType
@PublicAPI
class DatasetReader(InputReader):
    """Reader object that loads data from Ray Dataset.

    Examples:
        config = {
            "input": "dataset",
            "input_config": {
                "format": "json",
                # A single data file, a directory, or anything
                # that ray.data.dataset recognizes.
                "paths": "/tmp/sample_batches/",
                # By default, parallelism=num_workers.
                "parallelism": 3,
                # Dataset allocates 0.5 CPU for each reader by default.
                # Adjust this value based on the size of your offline dataset.
                "num_cpus_per_read_task": 0.5,
            }
        }
    """

    @PublicAPI
    def __init__(self, ds: ray.data.Dataset, ioctx: Optional[IOContext]=None):
        """Initializes a DatasetReader instance.

        Args:
            ds: Ray dataset to sample from.
        """
        self._ioctx = ioctx or IOContext()
        self._default_policy = self.policy_map = None
        self.preprocessor = None
        self._dataset = ds
        self.count = None if not self._dataset else self._dataset.count()
        ray.data.set_progress_bars(enabled=False)
        self.batch_size = self._ioctx.config.get('train_batch_size', 1)
        num_workers = self._ioctx.config.get('num_workers', 0)
        seed = self._ioctx.config.get('seed', None)
        if num_workers:
            self.batch_size = max(math.ceil(self.batch_size / num_workers), 1)
        if ds:
            if self._ioctx.worker is not None:
                self._policy_map = self._ioctx.worker.policy_map
                self._default_policy = self._policy_map.get(DEFAULT_POLICY_ID)
                self.preprocessor = self._ioctx.worker.preprocessors.get(DEFAULT_POLICY_ID) if not self._ioctx.config.get('_disable_preprocessors', False) else None
            print(f'DatasetReader {self._ioctx.worker_index} has {ds.count()}, samples.')

            def iterator():
                while True:
                    ds = self._dataset.random_shuffle(seed=seed)
                    yield from ds.iter_rows()
            self._iter = iterator()
        else:
            self._iter = None

    @override(InputReader)
    def next(self) -> SampleBatchType:
        assert self._iter is not None
        ret = []
        count = 0
        while count < self.batch_size:
            d = next(self._iter)
            d = from_json_data(d, self._ioctx.worker)
            count += d.count
            d = self._preprocess_if_needed(d)
            d = postprocess_actions(d, self._ioctx)
            d = self._postprocess_if_needed(d)
            ret.append(d)
        ret = concat_samples(ret)
        return ret

    def _preprocess_if_needed(self, batch: SampleBatchType) -> SampleBatchType:
        if self.preprocessor:
            for key in (SampleBatch.CUR_OBS, SampleBatch.NEXT_OBS):
                if key in batch:
                    batch[key] = np.stack([self.preprocessor.transform(s) for s in batch[key]])
        return batch

    def _postprocess_if_needed(self, batch: SampleBatchType) -> SampleBatchType:
        if not self._ioctx.config.get('postprocess_inputs'):
            return batch
        if isinstance(batch, SampleBatch):
            out = []
            for sub_batch in batch.split_by_episode():
                if self._default_policy is not None:
                    out.append(self._default_policy.postprocess_trajectory(sub_batch))
                else:
                    out.append(sub_batch)
            return concat_samples(out)
        else:
            raise NotImplementedError('Postprocessing of multi-agent data not implemented yet.')