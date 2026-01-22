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
def _interact(self, location, error_info, payload):
    """Gathers a macaroon by directing the user to interact with a
        web page. The error_info argument holds the interaction-required
        error response.
        @return DischargeToken, bakery.Macaroon
        """
    if self._interaction_methods is None or len(self._interaction_methods) == 0:
        raise InteractionError('interaction required but not possible')
    if error_info.info.interaction_methods is None and error_info.info.visit_url is not None:
        return (None, self._legacy_interact(location, error_info))
    for interactor in self._interaction_methods:
        found = error_info.info.interaction_methods.get(interactor.kind())
        if found is None:
            continue
        try:
            token = interactor.interact(self, location, error_info)
        except InteractionMethodNotFound:
            continue
        if token is None:
            raise InteractionError('interaction method returned an empty token')
        return (token, None)
    raise InteractionError('no supported interaction method')