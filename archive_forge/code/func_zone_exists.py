from oslo_log import log as logging
from os_win._i18n import _
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
def zone_exists(self, zone_name):
    return self._get_zone(zone_name) is not None