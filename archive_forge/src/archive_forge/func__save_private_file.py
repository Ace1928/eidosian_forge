import collections
import copy
import datetime
import json
import logging
import os
import shutil
import socket
import sys
import tempfile
import six
from six.moves import http_client
from six.moves import urllib
import oauth2client
from oauth2client import _helpers
from oauth2client import _pkce
from oauth2client import clientsecrets
from oauth2client import transport
def _save_private_file(filename, json_contents):
    """Saves a file with read-write permissions on for the owner.

    Args:
        filename: String. Absolute path to file.
        json_contents: JSON serializable object to be saved.
    """
    temp_filename = tempfile.mktemp()
    file_desc = os.open(temp_filename, os.O_WRONLY | os.O_CREAT, 384)
    with os.fdopen(file_desc, 'w') as file_handle:
        json.dump(json_contents, file_handle, sort_keys=True, indent=2, separators=(',', ': '))
    shutil.move(temp_filename, filename)