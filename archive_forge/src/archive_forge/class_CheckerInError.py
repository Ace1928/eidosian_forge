import json
import logging
import os
import tempfile
from datetime import datetime, timedelta
from unittest import TestCase
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
import macaroonbakery.httpbakery as httpbakery
import macaroonbakery.httpbakery.agent as agent
import requests.cookies
from httmock import HTTMock, response, urlmatch
from six.moves.urllib.parse import parse_qs, urlparse
class CheckerInError(bakery.ThirdPartyCaveatChecker):

    def check_third_party_caveat(self, ctx, info):
        InfoStorage.info = info
        raise InteractionRequiredError(httpbakery.Error(code=httpbakery.ERR_INTERACTION_REQUIRED, version=httpbakery.request_version(request.headers), message='interaction required', info=httpbakery.ErrorInfo(wait_url='http://0.3.2.1/wait?dischargeid=1', visit_url='http://0.3.2.1/visit?dischargeid=1')))