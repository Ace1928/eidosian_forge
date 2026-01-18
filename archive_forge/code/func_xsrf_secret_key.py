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
def xsrf_secret_key():
    """Return the secret key for use for XSRF protection.

    If the Site entity does not have a secret key, this method will also create
    one and persist it.

    Returns:
        The secret key.
    """
    secret = memcache.get(XSRF_MEMCACHE_ID, namespace=OAUTH2CLIENT_NAMESPACE)
    if not secret:
        model = SiteXsrfSecretKey.get_or_insert(key_name='site')
        if not model.secret:
            model.secret = _generate_new_xsrf_secret_key()
            model.put()
        secret = model.secret
        memcache.add(XSRF_MEMCACHE_ID, secret, namespace=OAUTH2CLIENT_NAMESPACE)
    return str(secret)