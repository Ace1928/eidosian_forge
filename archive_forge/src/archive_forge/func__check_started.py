from os_ken.services.protocols.bgp.base import Activity
from os_ken.services.protocols.bgp.base import ActivityException
from os_ken.services.protocols.bgp.rtconf.neighbors import NeighborsConf
from os_ken.services.protocols.bgp.rtconf.vrfs import VrfsConf
def _check_started(self):
    if not self.started:
        raise ActivityException('Cannot access any property before activity has started')