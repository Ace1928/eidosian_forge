import time
from io import BytesIO
from ... import errors as bzr_errors
from ... import tests
from ...tests.features import Feature, ModuleAvailableFeature
from .. import import_dulwich
def _create_blob(self, content):
    self._counter += 1
    from fastimport.commands import BlobCommand
    blob = BlobCommand(b'%d' % self._counter, content)
    self._write(bytes(blob) + b'\n')
    return self._counter