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
class MyInteractor(httpbakery.LegacyInteractor):

    def legacy_interact(self, ctx, location, visit_url):
        raise httpbakery.InteractionError('cannot visit')

    def interact(self, ctx, location, interaction_required_err):
        pass

    def kind(self):
        return httpbakery.WEB_BROWSER_INTERACTION_KIND