import functools
import pathlib
from contextlib import suppress
from types import SimpleNamespace
from .. import readers, _adapters
def _namespace_reader(self):
    with suppress(AttributeError, ValueError):
        return readers.NamespaceReader(self.spec.submodule_search_locations)