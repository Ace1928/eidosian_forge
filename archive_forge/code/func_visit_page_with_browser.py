import base64
import binascii
import ipaddress
import json
import webbrowser
from datetime import datetime
import six
from pymacaroons import Macaroon
from pymacaroons.serializers import json_serializer
import six.moves.http_cookiejar as http_cookiejar
from six.moves.urllib.parse import urlparse
def visit_page_with_browser(visit_url):
    """Open a browser so the user can validate its identity.

    @param visit_url: where to prove your identity.
    """
    webbrowser.open(visit_url, new=1)
    print('Opening an authorization web page in your browser.')
    print('If it does not open, please open this URL:\n', visit_url, '\n')