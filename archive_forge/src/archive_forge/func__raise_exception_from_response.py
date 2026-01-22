from __future__ import annotations
import time
from typing import NoReturn
from ._version import VERSION
from ._exceptions import APIException
from .actions import ActionsClient
from .certificates import CertificatesClient
from .datacenters import DatacentersClient
from .firewalls import FirewallsClient
from .floating_ips import FloatingIPsClient
from .images import ImagesClient
from .isos import IsosClient
from .load_balancer_types import LoadBalancerTypesClient
from .load_balancers import LoadBalancersClient
from .locations import LocationsClient
from .networks import NetworksClient
from .placement_groups import PlacementGroupsClient
from .primary_ips import PrimaryIPsClient
from .server_types import ServerTypesClient
from .servers import ServersClient
from .ssh_keys import SSHKeysClient
from .volumes import VolumesClient
def _raise_exception_from_response(self, response) -> NoReturn:
    raise APIException(code=response.status_code, message=response.reason, details={'content': response.content})