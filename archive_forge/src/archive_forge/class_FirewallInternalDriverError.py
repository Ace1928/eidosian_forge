from neutron_lib._i18n import _
from neutron_lib import exceptions
class FirewallInternalDriverError(exceptions.NeutronException):
    """FWaaS exception for all driver errors

    On any failure or exception in the driver, driver should log it and
    raise this exception to the agent
    """
    message = _('%(driver)s: Internal driver error.')