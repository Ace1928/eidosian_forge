from typing import Any, Dict, Optional
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.compute.base import VolumeSnapshot
class VultrNetwork:
    """
    Represents information about a Vultr private network.
    """

    def __init__(self, id: str, cidr_block: str, location: str, extra: Optional[Dict[str, Any]]=None) -> None:
        self.id = id
        self.cidr_block = cidr_block
        self.location = location
        self.extra = extra or {}

    def __repr__(self):
        return '<Vultrnetwork: id={} cidr_block={} location={}>'.format(self.id, self.cidr_block, self.location)