import base64
import json
import logging
import os
import threading
import fasteners
from six import iteritems
from oauth2client import _helpers
from oauth2client import client
def _refresh_predicate(self, credentials):
    if credentials is None:
        return True
    elif credentials.invalid:
        return True
    elif credentials.access_token_expired:
        return True
    else:
        return False