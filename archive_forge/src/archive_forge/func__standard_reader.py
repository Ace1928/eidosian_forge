import functools
import pathlib
from contextlib import suppress
from types import SimpleNamespace
from .. import readers, _adapters
def _standard_reader(self):
    return self._zip_reader() or self._namespace_reader() or self._file_reader()