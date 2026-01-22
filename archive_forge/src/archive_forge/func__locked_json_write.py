import errno
import json
import logging
import os
import threading
from oauth2client import client
from oauth2client import util
from oauth2client.contrib import locked_file
def _locked_json_write(self, data):
    """Write a JSON serializable data structure to the multistore.

        The multistore must be locked when this is called.

        Args:
            data: The data to be serialized and written.
        """
    assert self._thread_lock.locked()
    if self._read_only:
        return
    self._file.file_handle().seek(0)
    json.dump(data, self._file.file_handle(), sort_keys=True, indent=2, separators=(',', ': '))
    self._file.file_handle().truncate()