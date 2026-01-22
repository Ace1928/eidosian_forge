from functools import wraps
import hashlib
import json
import os
import pickle
import six.moves.http_client as httplib
from oauth2client import client
from oauth2client import clientsecrets
from oauth2client import transport
from oauth2client.contrib import dictionary_storage
def _create_blueprint(self):
    bp = Blueprint('oauth2', __name__)
    bp.add_url_rule('/oauth2authorize', 'authorize', self.authorize_view)
    bp.add_url_rule('/oauth2callback', 'callback', self.callback_view)
    return bp