import base64
from collections import namedtuple
import requests
from ._error import InteractionError
from ._interactor import (
from macaroonbakery._utils import visit_page_with_browser
from six.moves.urllib.parse import urljoin
def _wait_for_token(self, ctx, wait_token_url):
    """ Returns a token from a the wait token URL
        @param wait_token_url URL to wait for (string)
        :return DischargeToken
        """
    resp = requests.get(wait_token_url)
    if resp.status_code != 200:
        raise InteractionError('cannot get {}'.format(wait_token_url))
    json_resp = resp.json()
    kind = json_resp.get('kind')
    if kind is None:
        raise InteractionError('cannot get kind token from {}'.format(wait_token_url))
    token_val = json_resp.get('token')
    if token_val is None:
        token_val = json_resp.get('token64')
        if token_val is None:
            raise InteractionError('cannot get token from {}'.format(wait_token_url))
        token_val = base64.b64decode(token_val)
    return DischargeToken(kind=kind, value=token_val)