import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def _get_outscale_endpoint(self, region: str, version: str, action: str):
    return 'https://api.{}.{}/api/{}/{}'.format(region, self.base_uri, version, action)