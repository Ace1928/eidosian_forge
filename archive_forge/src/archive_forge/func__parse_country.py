import json
import base64
from libcloud.utils.py3 import b, httplib
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeState, NodeDriver, NodeLocation
from libcloud.compute.types import Provider
from libcloud.common.upcloud import (
def _parse_country(self, zone_id):
    """Parses the country information out of zone_id.
        Zone_id format [country]_[city][number], like fi_hel1"""
    return zone_id.split('-')[0].upper()