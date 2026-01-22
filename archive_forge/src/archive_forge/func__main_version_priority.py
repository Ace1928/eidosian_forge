import time
import random
from os_ken.base import app_manager
from os_ken.lib import hub
from os_ken.lib import mac as lib_mac
from os_ken.lib.packet import vrrp
from os_ken.services.protocols.vrrp import api as vrrp_api
from os_ken.services.protocols.vrrp import event as vrrp_event
def _main_version_priority(self, vrrp_version, priority):
    self._main_version_priority_sleep(vrrp_version, priority, False)
    self._main_version_priority_sleep(vrrp_version, priority, True)