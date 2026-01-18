import calendar
import copy
import http.cookiejar as http_cookiejar
from http.cookies import SimpleCookie
import logging
import re
import time
from urllib.parse import urlencode
from urllib.parse import urlparse
import requests
from saml2 import SAMLError
from saml2 import class_name
from saml2.pack import make_soap_enveloped_saml_thingy
from saml2.time_util import utc_now
@staticmethod
def use_http_artifact(message, destination='', relay_state=''):
    if relay_state:
        query = urlencode({'SAMLart': message, 'RelayState': relay_state})
    else:
        query = urlencode({'SAMLart': message})
    info = {'data': '', 'url': f'{destination}?{query}'}
    return info