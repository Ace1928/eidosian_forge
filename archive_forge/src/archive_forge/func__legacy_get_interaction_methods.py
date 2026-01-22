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
def _legacy_get_interaction_methods(u):
    """ Queries a URL as found in an ErrInteractionRequired VisitURL field to
    find available interaction methods.
    It does this by sending a GET request to the URL with the Accept
    header set to "application/json" and parsing the resulting
    response as a dict.
    """
    headers = {BAKERY_PROTOCOL_HEADER: str(bakery.LATEST_VERSION), 'Accept': 'application/json'}
    resp = requests.get(url=u, headers=headers)
    method_urls = {}
    if resp.status_code == 200:
        json_resp = resp.json()
        for m in json_resp:
            method_urls[m] = urljoin(u, json_resp[m])
    if method_urls.get('interactive') is None:
        method_urls['interactive'] = u
    return method_urls