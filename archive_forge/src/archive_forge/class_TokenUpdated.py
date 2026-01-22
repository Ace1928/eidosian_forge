import logging
from oauthlib.common import generate_token, urldecode
from oauthlib.oauth2 import WebApplicationClient, InsecureTransportError
from oauthlib.oauth2 import LegacyApplicationClient
from oauthlib.oauth2 import TokenExpiredError, is_secure_transport
import requests
class TokenUpdated(Warning):

    def __init__(self, token):
        super(TokenUpdated, self).__init__()
        self.token = token