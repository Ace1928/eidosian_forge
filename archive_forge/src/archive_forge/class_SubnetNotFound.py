from oslo_utils import excutils
from neutron_lib._i18n import _
class SubnetNotFound(NotFound):
    """An exception for a requested subnet that's not found.

    A specialization of the NotFound exception indicating a requested subnet
    could not be found.

    :param subnet_id: The UUID of the (not found) subnet that was requested.
    """
    message = _('Subnet %(subnet_id)s could not be found.')