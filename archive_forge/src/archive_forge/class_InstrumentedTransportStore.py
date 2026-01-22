import gzip
import os
from io import BytesIO
from ... import errors as errors
from ... import transactions, transport
from ...bzr.weave import WeaveFile
from ...errors import BzrError
from ...tests import TestCase, TestCaseInTempDir, TestCaseWithTransport
from ...transport.memory import MemoryTransport
from .store import TransportStore
from .store.text import TextStore
from .store.versioned import VersionedFileStore
class InstrumentedTransportStore(TransportStore):
    """An instrumented TransportStore.

    Here we replace template method worker methods with calls that record the
    expected results.
    """

    def _add(self, filename, file):
        self._calls.append(('_add', filename, file))

    def __init__(self, transport, prefixed=False):
        super().__init__(transport, prefixed)
        self._calls = []