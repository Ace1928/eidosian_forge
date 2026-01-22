import logging
from google.appengine.ext import ndb
from oauth2client import client
@classmethod
def _get_kind(cls):
    """Return the kind name for this class."""
    return 'CredentialsModel'