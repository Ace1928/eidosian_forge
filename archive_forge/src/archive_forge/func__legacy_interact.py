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
def _legacy_interact(self, location, error_info):
    visit_url = urljoin(location, error_info.info.visit_url)
    wait_url = urljoin(location, error_info.info.wait_url)
    method_urls = {'interactive': visit_url}
    if len(self._interaction_methods) > 1 or self._interaction_methods[0].kind() != WEB_BROWSER_INTERACTION_KIND:
        method_urls = _legacy_get_interaction_methods(visit_url)
    for interactor in self._interaction_methods:
        kind = interactor.kind()
        if kind == WEB_BROWSER_INTERACTION_KIND:
            kind = 'interactive'
        if not isinstance(interactor, LegacyInteractor):
            continue
        visit_url = method_urls.get(kind)
        if visit_url is None:
            continue
        visit_url = urljoin(location, visit_url)
        interactor.legacy_interact(self, location, visit_url)
        return _wait_for_macaroon(wait_url)
    raise InteractionError('no methods supported; supported [{}]; provided [{}]'.format(' '.join([x.kind() for x in self._interaction_methods]), ' '.join(method_urls.keys())))