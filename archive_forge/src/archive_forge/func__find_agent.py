import copy
import json
import logging
from collections import namedtuple
import macaroonbakery.bakery as bakery
import macaroonbakery.httpbakery as httpbakery
import macaroonbakery._utils as utils
import requests.cookies
from six.moves.urllib.parse import urljoin
def _find_agent(self, location):
    """ Finds an appropriate agent entry for the given location.
        :return Agent
        """
    for a in self._auth_info.agents:
        if a.url.rstrip('/') == location.rstrip('/'):
            return a
    raise httpbakery.InteractionMethodNotFound('cannot find username for discharge location {}'.format(location))