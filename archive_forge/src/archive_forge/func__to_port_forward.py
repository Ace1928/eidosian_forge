import json
from libcloud.compute.base import (
from libcloud.common.gig_g8 import G8Connection
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def _to_port_forward(self, data, ex_network):
    return G8PortForward(ex_network, str(data['machineId']), data['publicPort'], data['localPort'], data['protocol'], self)