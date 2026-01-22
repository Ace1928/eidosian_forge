import os
from ... import recordio, ndarray
class RecordFileDataset(Dataset):
    """A dataset wrapping over a RecordIO (.rec) file.

    Each sample is a string representing the raw content of an record.

    Parameters
    ----------
    filename : str
        Path to rec file.
    """

    def __init__(self, filename):
        self.idx_file = os.path.splitext(filename)[0] + '.idx'
        self.filename = filename
        self._record = recordio.MXIndexedRecordIO(self.idx_file, self.filename, 'r')

    def __getitem__(self, idx):
        return self._record.read_idx(self._record.keys[idx])

    def __len__(self):
        return len(self._record.keys)