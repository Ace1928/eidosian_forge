import errno
import json
import logging
import os
import threading
from oauth2client import client
from oauth2client import util
from oauth2client.contrib import locked_file
def _get_all_credential_keys(self):
    """Gets all the registered credential keys in the multistore.

        Returns:
            A list of dictionaries corresponding to all the keys currently
            registered
        """
    return [dict(key) for key in self._data.keys()]