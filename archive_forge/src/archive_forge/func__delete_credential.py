import errno
import json
import logging
import os
import threading
from oauth2client import client
from oauth2client import util
from oauth2client.contrib import locked_file
def _delete_credential(self, key):
    """Delete a credential and write the multistore.

        This must be called when the multistore is locked.

        Args:
            key: The key used to retrieve the credential
        """
    try:
        del self._data[key]
    except KeyError:
        pass
    self._write()