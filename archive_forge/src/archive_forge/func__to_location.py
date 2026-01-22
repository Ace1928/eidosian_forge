import json
from typing import List
from datetime import datetime
import requests
from libcloud.common.osc import OSCRequestSignerAlgorithmV4
from libcloud.common.base import ConnectionUserAndKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def _to_location(self, location):
    country = location['Name'].split(', ')[1]
    return NodeLocation(id=location['Code'], name=location['Name'], country=country, driver=self, extra=location)