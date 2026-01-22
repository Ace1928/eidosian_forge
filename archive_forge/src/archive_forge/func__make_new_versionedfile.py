import os
from .... import errors, osutils
from .... import transport as _mod_transport
from .... import ui
from ....trace import mutter
from . import TransportStore
def _make_new_versionedfile(self, file_id, transaction, known_missing=False, _filename=None):
    """Make a new versioned file.

        :param _filename: filename that would be returned from self.filename for
        file_id. This is used to reduce duplicate filename calculations when
        using 'get_weave_or_empty'. FOR INTERNAL USE ONLY.
        """
    if not known_missing and self.has_id(file_id):
        self.delete(file_id, transaction)
    if _filename is None:
        _filename = self.filename(file_id)
    try:
        weave = self._versionedfile_class(_filename, self._transport, self._file_mode, create=True, get_scope=self.get_scope, **self._versionedfile_kwargs)
    except _mod_transport.NoSuchFile:
        if not self._prefixed:
            raise
        dirname = osutils.dirname(_filename)
        self._transport.mkdir(dirname, mode=self._dir_mode)
        weave = self._versionedfile_class(_filename, self._transport, self._file_mode, create=True, get_scope=self.get_scope, **self._versionedfile_kwargs)
    return weave