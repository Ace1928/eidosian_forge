import os
from ... import recordio, ndarray
class _LazyTransformDataset(Dataset):
    """Lazily transformed dataset."""

    def __init__(self, data, fn):
        self._data = data
        self._fn = fn

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        item = self._data[idx]
        if isinstance(item, tuple):
            return self._fn(*item)
        return self._fn(item)