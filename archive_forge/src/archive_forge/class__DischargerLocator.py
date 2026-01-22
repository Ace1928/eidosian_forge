import base64
import datetime
import json
import platform
import threading
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
import macaroonbakery.httpbakery as httpbakery
import pymacaroons
import requests
import macaroonbakery._utils as utils
from macaroonbakery.httpbakery._error import DischargeError
from fixtures import (
from httmock import HTTMock, urlmatch
from six.moves.urllib.parse import parse_qs
from six.moves.urllib.request import Request
class _DischargerLocator(bakery.ThirdPartyLocator):

    def __init__(self):
        self.key = bakery.generate_key()

    def third_party_info(self, loc):
        if loc == 'http://1.2.3.4':
            return bakery.ThirdPartyInfo(public_key=self.key.public_key, version=bakery.LATEST_VERSION)