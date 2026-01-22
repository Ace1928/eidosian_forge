import os
from .... import errors, osutils
from .... import transport as _mod_transport
from .... import ui
from ....trace import mutter
from . import TransportStore
def _put_weave(self, file_id, weave, transaction):
    """Preserved here for upgrades-to-weaves to use."""
    myweave = self._make_new_versionedfile(file_id, transaction)
    myweave.insert_record_stream(weave.get_record_stream([(version,) for version in weave.versions()], 'topological', False))