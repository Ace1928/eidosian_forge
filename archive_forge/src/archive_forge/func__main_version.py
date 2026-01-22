import time
import random
from os_ken.base import app_manager
from os_ken.lib import hub
from os_ken.lib import mac as lib_mac
from os_ken.lib.packet import vrrp
from os_ken.services.protocols.vrrp import api as vrrp_api
from os_ken.services.protocols.vrrp import event as vrrp_event
def _main_version(self, vrrp_version):
    self._main_version_priority(vrrp_version, vrrp.VRRP_PRIORITY_ADDRESS_OWNER)
    self._main_version_priority(vrrp_version, vrrp.VRRP_PRIORITY_BACKUP_MAX)
    self._main_version_priority(vrrp_version, vrrp.VRRP_PRIORITY_BACKUP_DEFAULT)
    self._main_version_priority(vrrp_version, vrrp.VRRP_PRIORITY_BACKUP_MIN)