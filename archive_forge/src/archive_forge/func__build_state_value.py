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
def _build_state_value(request_handler, user):
    """Composes the value for the 'state' parameter.

    Packs the current request URI and an XSRF token into an opaque string that
    can be passed to the authentication server via the 'state' parameter.

    Args:
        request_handler: webapp.RequestHandler, The request.
        user: google.appengine.api.users.User, The current user.

    Returns:
        The state value as a string.
    """
    uri = request_handler.request.url
    token = xsrfutil.generate_token(xsrf_secret_key(), user.user_id(), action_id=str(uri))
    return uri + ':' + token