from datetime import datetime
from libcloud.common.gandi import (
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def _to_ifaces(self, ifaces):
    return [self._to_iface(i) for i in ifaces]