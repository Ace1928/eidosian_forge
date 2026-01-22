import functools
import pathlib
from contextlib import suppress
from types import SimpleNamespace
from .. import readers, _adapters
def _zip_reader(self):
    with suppress(AttributeError):
        return readers.ZipReader(self.spec.loader, self.spec.name)