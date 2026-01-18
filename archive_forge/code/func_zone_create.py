from oslo_log import log as logging
from os_win._i18n import _
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
def zone_create(self, zone_name, zone_type, ds_integrated, data_file_name=None, ip_addrs=None, admin_email_name=None):
    """Creates a DNS Zone and returns the path to the associated object.

        :param zone_name: string representing the name of the zone.
        :param zone_type: type of zone
            0 = Primary zone
            1 = Secondary zone, MUST include at least one master IP
            2 = Stub zone, MUST include at least one master IP
            3 = Zone forwarder, MUST include at least one master IP
        :param ds_integrated: Only Primary zones can be stored in AD
            True = the zone data is stored in the Active Directory
            False = the data zone is stored in files
        :param data_file_name(Optional): name of the data file associated
            with the zone.
        :param ip_addrs(Optional): IP addresses of the master DNS servers
            for this zone. Parameter type MUST be list
        :param admin_email_name(Optional): email address of the administrator
            responsible for the zone.
        """
    LOG.debug("Creating DNS Zone '%s'" % zone_name)
    if self.zone_exists(zone_name):
        raise exceptions.DNSZoneAlreadyExists(zone_name=zone_name)
    dns_zone_manager = self._dns_manager.MicrosoftDNS_Zone
    zone_path, = dns_zone_manager.CreateZone(ZoneName=zone_name, ZoneType=zone_type, DsIntegrated=ds_integrated, DataFileName=data_file_name, IpAddr=ip_addrs, AdminEmailname=admin_email_name)
    return zone_path