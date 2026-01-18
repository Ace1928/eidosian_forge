from oslo_log import log as logging
from os_win._i18n import _
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
def zone_update(self, zone_name):
    LOG.debug("Updating DNS Zone '%s'" % zone_name)
    zone = self._get_zone(zone_name, ignore_missing=False)
    if zone.DsIntegrated and zone.ZoneType == constants.DNS_ZONE_TYPE_PRIMARY:
        zone.UpdateFromDS()
    elif zone.ZoneType in [constants.DNS_ZONE_TYPE_SECONDARY, constants.DNS_ZONE_TYPE_STUB]:
        zone.ForceRefresh()
    elif zone.ZoneType in [constants.DNS_ZONE_TYPE_PRIMARY, constants.DNS_ZONE_TYPE_FORWARD]:
        zone.ReloadZone()