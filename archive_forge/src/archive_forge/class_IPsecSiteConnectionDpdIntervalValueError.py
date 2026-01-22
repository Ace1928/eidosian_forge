from neutron_lib._i18n import _
from neutron_lib import exceptions
class IPsecSiteConnectionDpdIntervalValueError(exceptions.InvalidInput):
    message = _('ipsec_site_connection %(attr)s is equal to or less than dpd_interval')