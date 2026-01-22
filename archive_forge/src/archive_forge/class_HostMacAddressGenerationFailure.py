from oslo_utils import excutils
from neutron_lib._i18n import _
class HostMacAddressGenerationFailure(ServiceUnavailable):
    """MAC address generation failure for a host.

    :param host: The host MAC address generation failed for.
    """
    message = _('Unable to generate unique mac for host %(host)s.')