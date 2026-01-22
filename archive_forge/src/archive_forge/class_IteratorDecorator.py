import inspect
import functools
from enum import Enum
import torch.autograd
class IteratorDecorator:
    """
        Wrap the iterator and modifying its `__next__` method.

        This decorator is applied to DataPipes of which `__iter__` method is NOT a generator function.
        Those `__iter__` method commonly returns `self` but not necessarily.
        """

    def __init__(self, iterator, datapipe, iterator_id, has_next_method):
        self.iterator = iterator
        self.datapipe = datapipe
        self.iterator_id = iterator_id
        self._profiler_enabled = torch.autograd._profiler_enabled()
        self.self_and_has_next_method = self.iterator is self.datapipe and has_next_method

    def __iter__(self):
        return self

    def _get_next(self):
        """Return next with logic related to iterator validity, profiler, and incrementation of samples yielded."""
        _check_iterator_valid(self.datapipe, self.iterator_id)
        result = next(self.iterator)
        if not self.self_and_has_next_method:
            self.datapipe._number_of_samples_yielded += 1
        return result

    def __next__(self):
        if self._profiler_enabled:
            with profiler_record_fn_context(self.datapipe):
                return self._get_next()
        else:
            return self._get_next()

    def __getattr__(self, name):
        return getattr(self.iterator, name)