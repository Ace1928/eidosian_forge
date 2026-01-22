from pathlib import Path
import json
from collections.abc import MutableMapping, Mapping
from contextlib import contextmanager
from ase.io.jsonio import read_json, write_json, encode
from ase.utils import opencew
class CombinedJSONCache(Mapping):
    writable = False

    def __init__(self, directory, dct):
        self.directory = Path(directory)
        self._dct = dict(dct)

    def filecount(self):
        return int(self._filename.is_file())

    @property
    def _filename(self):
        return self.directory / 'combined.json'

    def _dump_json(self):
        target = self._filename
        if target.exists():
            raise RuntimeError(f'Already exists: {target}')
        self.directory.mkdir(exist_ok=True, parents=True)
        write_json(target, self._dct)

    def __len__(self):
        return len(self._dct)

    def __iter__(self):
        return iter(self._dct)

    def __getitem__(self, index):
        return self._dct[index]

    @classmethod
    def dump_cache(cls, path, dct):
        cache = cls(path, dct)
        cache._dump_json()
        return cache

    @classmethod
    def load(cls, path):
        cache = cls(path, {})
        dct = read_json(cache._filename, always_array=False)
        cache._dct.update(dct)
        return cache

    def clear(self):
        self._filename.unlink()
        self._dct.clear()

    def combine(self):
        return self

    def split(self):
        cache = MultiFileJSONCache(self.directory)
        assert len(cache) == 0
        cache.update(self)
        assert set(cache) == set(self)
        self.clear()
        return cache