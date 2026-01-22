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
def _prepare_discharge_hook(req, client):
    """ Return the hook function (called when the response is received.)

    This allows us to intercept the response and do any necessary
    macaroon discharge before returning.
    """

    class Retry:
        count = 0

    def hook(response, *args, **kwargs):
        """ Requests hooks system, this is the hook for the response.
        """
        status_code = response.status_code
        if status_code != 407 and status_code != 401:
            return response
        if status_code == 401 and response.headers.get('WWW-Authenticate') != 'Macaroon':
            return response
        if response.headers.get('Content-Type') != 'application/json':
            return response
        errorJSON = response.json()
        if errorJSON.get('Code') != ERR_DISCHARGE_REQUIRED:
            return response
        error = Error.from_dict(errorJSON)
        Retry.count += 1
        if Retry.count >= MAX_DISCHARGE_RETRIES:
            raise BakeryException('too many ({}) discharge requests'.format(Retry.count))
        client.handle_error(error, req.url)
        req.headers.pop('Cookie', None)
        req.prepare_cookies(client.cookies)
        req.headers[BAKERY_PROTOCOL_HEADER] = str(bakery.LATEST_VERSION)
        with requests.Session() as s:
            settings = s.merge_environment_settings(req.url, {}, None, None, None)
            return s.send(req, **settings)
    return hook