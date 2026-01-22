import gc
import os
import warnings
import threading
import contextlib
from abc import ABCMeta, abstractmethod
from ._multiprocessing_helpers import mp
class AutoBatchingMixin(object):
    """A helper class for automagically batching jobs."""
    MIN_IDEAL_BATCH_DURATION = 0.2
    MAX_IDEAL_BATCH_DURATION = 2
    _DEFAULT_EFFECTIVE_BATCH_SIZE = 1
    _DEFAULT_SMOOTHED_BATCH_DURATION = 0.0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._effective_batch_size = self._DEFAULT_EFFECTIVE_BATCH_SIZE
        self._smoothed_batch_duration = self._DEFAULT_SMOOTHED_BATCH_DURATION

    def compute_batch_size(self):
        """Determine the optimal batch size"""
        old_batch_size = self._effective_batch_size
        batch_duration = self._smoothed_batch_duration
        if batch_duration > 0 and batch_duration < self.MIN_IDEAL_BATCH_DURATION:
            ideal_batch_size = int(old_batch_size * self.MIN_IDEAL_BATCH_DURATION / batch_duration)
            ideal_batch_size *= 2
            batch_size = min(2 * old_batch_size, ideal_batch_size)
            batch_size = max(batch_size, 1)
            self._effective_batch_size = batch_size
            if self.parallel.verbose >= 10:
                self.parallel._print(f'Batch computation too fast ({batch_duration}s.) Setting batch_size={batch_size}.')
        elif batch_duration > self.MAX_IDEAL_BATCH_DURATION and old_batch_size >= 2:
            ideal_batch_size = int(old_batch_size * self.MIN_IDEAL_BATCH_DURATION / batch_duration)
            batch_size = max(2 * ideal_batch_size, 1)
            self._effective_batch_size = batch_size
            if self.parallel.verbose >= 10:
                self.parallel._print(f'Batch computation too slow ({batch_duration}s.) Setting batch_size={batch_size}.')
        else:
            batch_size = old_batch_size
        if batch_size != old_batch_size:
            self._smoothed_batch_duration = self._DEFAULT_SMOOTHED_BATCH_DURATION
        return batch_size

    def batch_completed(self, batch_size, duration):
        """Callback indicate how long it took to run a batch"""
        if batch_size == self._effective_batch_size:
            old_duration = self._smoothed_batch_duration
            if old_duration == self._DEFAULT_SMOOTHED_BATCH_DURATION:
                new_duration = duration
            else:
                new_duration = 0.8 * old_duration + 0.2 * duration
            self._smoothed_batch_duration = new_duration

    def reset_batch_stats(self):
        """Reset batch statistics to default values.

        This avoids interferences with future jobs.
        """
        self._effective_batch_size = self._DEFAULT_EFFECTIVE_BATCH_SIZE
        self._smoothed_batch_duration = self._DEFAULT_SMOOTHED_BATCH_DURATION