from oslo_utils import excutils
from neutron_lib._i18n import _
class AdminRequired(NotAuthorized):
    """A not authorized exception indicating an admin is required.

    A specialization of the NotAuthorized exception that indicates and admin
    is required to carry out the operation or access a resource.

    :param reason: A message indicating additional details on why admin is
        required for the operation access.
    """
    message = _('User does not have admin privileges: %(reason)s.')