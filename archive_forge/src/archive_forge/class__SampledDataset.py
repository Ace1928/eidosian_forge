import os
from ... import recordio, ndarray
class _SampledDataset(Dataset):
    """Dataset with elements chosen by a sampler"""

    def __init__(self, dataset, sampler):
        self._dataset = dataset
        self._sampler = sampler
        self._indices = list(iter(sampler))

    def __len__(self):
        return len(self._sampler)

    def __getitem__(self, idx):
        return self._dataset[self._indices[idx]]