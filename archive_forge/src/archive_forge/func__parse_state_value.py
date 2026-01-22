import cgi
import json
import logging
import os
import pickle
import threading
from google.appengine.api import app_identity
from google.appengine.api import memcache
from google.appengine.api import users
from google.appengine.ext import db
from google.appengine.ext.webapp.util import login_required
import webapp2 as webapp
import oauth2client
from oauth2client import _helpers
from oauth2client import client
from oauth2client import clientsecrets
from oauth2client import transport
from oauth2client.contrib import xsrfutil
def _parse_state_value(state, user):
    """Parse the value of the 'state' parameter.

    Parses the value and validates the XSRF token in the state parameter.

    Args:
        state: string, The value of the state parameter.
        user: google.appengine.api.users.User, The current user.

    Returns:
        The redirect URI, or None if XSRF token is not valid.
    """
    uri, token = state.rsplit(':', 1)
    if xsrfutil.validate_token(xsrf_secret_key(), token, user.user_id(), action_id=uri):
        return uri
    else:
        return None