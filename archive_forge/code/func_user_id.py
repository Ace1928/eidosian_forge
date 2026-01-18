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
@property
def user_id(self):
    """Returns the a unique identifier for the user

        Returns None if there are no credentials.

        The id is provided by the current credentials' id_token.
        """
    if not self.credentials:
        return None
    try:
        return self.credentials.id_token['sub']
    except KeyError:
        current_app.logger.error('Invalid id_token {0}'.format(self.credentials.id_token))