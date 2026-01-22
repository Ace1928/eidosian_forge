import numpy as np
class FilterSampler(Sampler):
    """Samples elements from a Dataset for which `fn` returns True.

    Parameters
    ----------
    fn : callable
        A callable function that takes a sample and returns a boolean
    dataset : Dataset
        The dataset to filter.
    """

    def __init__(self, fn, dataset):
        self._fn = fn
        self._dataset = dataset
        self._indices = [i for i, sample in enumerate(dataset) if fn(sample)]

    def __iter__(self):
        return iter(self._indices)

    def __len__(self):
        return len(self._indices)