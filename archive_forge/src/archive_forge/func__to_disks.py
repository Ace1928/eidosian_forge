from datetime import datetime
from libcloud.common.gandi import (
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
def _to_disks(self, elements):
    return [self._to_disk(el) for el in elements]