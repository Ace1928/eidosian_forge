import copy
import json
import logging
from collections import namedtuple
import macaroonbakery.bakery as bakery
import macaroonbakery.httpbakery as httpbakery
import macaroonbakery._utils as utils
import requests.cookies
from six.moves.urllib.parse import urljoin
class AuthInfo(namedtuple('AuthInfo', 'key, agents')):
    """ Holds the agent information required to set up agent authentication
    information.

    It holds the agent's private key and information about the username
    associated with each known agent-authentication server.
    @param key the agent's private key (bakery.PrivateKey).
    @param agents information about the known agents (list of Agent).
    """