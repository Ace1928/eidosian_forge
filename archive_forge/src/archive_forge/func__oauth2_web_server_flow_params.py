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
def _oauth2_web_server_flow_params(kwargs):
    """Configures redirect URI parameters for OAuth2WebServerFlow."""
    params = {'access_type': 'offline', 'response_type': 'code'}
    params.update(kwargs)
    approval_prompt = params.get('approval_prompt')
    if approval_prompt is not None:
        logger.warning('The approval_prompt parameter for OAuth2WebServerFlow is deprecated. Please use the prompt parameter instead.')
        if approval_prompt == 'force':
            logger.warning('approval_prompt="force" has been adjusted to prompt="consent"')
            params['prompt'] = 'consent'
            del params['approval_prompt']
    return params