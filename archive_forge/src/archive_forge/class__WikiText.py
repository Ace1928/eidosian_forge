import os
import zipfile
import shutil
import numpy as np
from . import _constants as C
from ...data import dataset
from ...utils import download, check_sha1, _get_repo_file_url
from ....contrib import text
from .... import nd, base
class _WikiText(_LanguageModelDataset):

    def _read_batch(self, filename):
        with open(filename, 'r', encoding='utf8') as fin:
            content = fin.read()
        self._build_vocab(content)
        raw_data = [line for line in [x.strip().split() for x in content.splitlines()] if line]
        for line in raw_data:
            line.append(C.EOS_TOKEN)
        raw_data = self.vocabulary.to_indices([x for line in raw_data for x in line if x])
        data = raw_data[0:-1]
        label = raw_data[1:]
        return (np.array(data, dtype=np.int32), np.array(label, dtype=np.int32))

    def _get_data(self):
        archive_file_name, archive_hash = self._archive_file
        data_file_name, data_hash = self._data_file[self._segment]
        path = os.path.join(self._root, data_file_name)
        if not os.path.exists(path) or not check_sha1(path, data_hash):
            namespace = 'gluon/dataset/' + self._namespace
            downloaded_file_path = download(_get_repo_file_url(namespace, archive_file_name), path=self._root, sha1_hash=archive_hash)
            with zipfile.ZipFile(downloaded_file_path, 'r') as zf:
                for member in zf.namelist():
                    filename = os.path.basename(member)
                    if filename:
                        dest = os.path.join(self._root, filename)
                        with zf.open(member) as source, open(dest, 'wb') as target:
                            shutil.copyfileobj(source, target)
        data, label = self._read_batch(path)
        seq_len_mult = len(data) // self._seq_len * self._seq_len
        self._data = nd.array(data, dtype=data.dtype)[:seq_len_mult].reshape((-1, self._seq_len))
        self._label = nd.array(label, dtype=label.dtype)[:seq_len_mult].reshape((-1, self._seq_len))

    def __getitem__(self, idx):
        return (self._data[idx], self._label[idx])

    def __len__(self):
        return len(self._label)