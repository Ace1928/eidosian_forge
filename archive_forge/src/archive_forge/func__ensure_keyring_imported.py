from __future__ import print_function
import httplib2
import json
import os
from select import select
import stat
from sys import stdin
import time
import webbrowser
from base64 import (
from six.moves.urllib.parse import parse_qs
from lazr.restfulclient.errors import HTTPError
from lazr.restfulclient.authorize.oauth import (
from launchpadlib import uris
@staticmethod
def _ensure_keyring_imported():
    """Ensure the keyring module is imported (postponing side effects).

        The keyring module initializes the environment-dependent backend at
        import time (nasty).  We want to avoid that initialization because it
        may do things like prompt the user to unlock their password store
        (e.g., KWallet).
        """
    if 'keyring' not in globals():
        global keyring
        import keyring
    if 'NoKeyringError' not in globals():
        global NoKeyringError
        try:
            from keyring.errors import NoKeyringError
        except ImportError:
            NoKeyringError = RuntimeError