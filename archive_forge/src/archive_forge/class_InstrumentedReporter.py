import os
from io import StringIO
from .. import delta as _mod_delta
from .. import revision as _mod_revision
from .. import tests
from ..bzr.inventorytree import InventoryTreeChange
class InstrumentedReporter:

    def __init__(self):
        self.calls = []

    def report(self, path, versioned, renamed, copied, modified, exe_change, kind):
        self.calls.append((path, versioned, renamed, copied, modified, exe_change, kind))