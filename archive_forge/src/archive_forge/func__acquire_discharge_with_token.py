import base64
import json
import logging
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
import macaroonbakery._utils as utils
from ._browser import WebBrowserInteractor
from ._error import (
from ._interactor import (
import requests
from six.moves.http_cookies import SimpleCookie
from six.moves.urllib.parse import urljoin
def _acquire_discharge_with_token(self, cav, payload, token):
    req = {}
    _add_json_binary_field(cav.caveat_id_bytes, req, 'id')
    if token is not None:
        _add_json_binary_field(token.value, req, 'token')
        req['token-kind'] = token.kind
    if payload is not None:
        req['caveat64'] = base64.urlsafe_b64encode(payload).rstrip(b'=').decode('utf-8')
    loc = cav.location
    if not loc.endswith('/'):
        loc += '/'
    target = urljoin(loc, 'discharge')
    headers = {BAKERY_PROTOCOL_HEADER: str(bakery.LATEST_VERSION)}
    return self.request('POST', target, data=req, headers=headers)