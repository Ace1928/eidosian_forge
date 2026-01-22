import json
from libcloud.compute.base import (
from libcloud.common.gig_g8 import G8Connection
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
@property
def _location(self):
    if self._location_data is None:
        self._location_data = self._api_request('/locations/list')[0]
    return self._location_data