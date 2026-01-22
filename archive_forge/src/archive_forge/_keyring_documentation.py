import macaroonbakery.bakery as bakery
import requests
from ._error import BAKERY_PROTOCOL_HEADER
from six.moves.urllib.parse import urlparse

        @param url: the url to retrieve public_key
        @param allow_insecure: By default it refuses to use insecure URLs.
        