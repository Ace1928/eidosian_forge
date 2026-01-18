from neutron_lib._i18n import _
from neutron_lib import exceptions
An operational error indicates that port still has an associated FIP.

    A specialization of the InUse exception indicating an operation failed on
    a port because it still has an associated FIP.

    :param port_id: The UUID of the port requested.
    