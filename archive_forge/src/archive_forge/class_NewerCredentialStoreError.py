import errno
import json
import logging
import os
import threading
from oauth2client import client
from oauth2client import util
from oauth2client.contrib import locked_file
class NewerCredentialStoreError(Error):
    """The credential store is a newer version than supported."""