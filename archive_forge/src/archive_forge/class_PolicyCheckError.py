from oslo_utils import excutils
from neutron_lib._i18n import _
class PolicyCheckError(NeutronException):
    """An error due to a policy check failure.

    :param policy: The policy that failed to check.
    :param reason: Additional details on the failure.
    """
    message = _('Failed to check policy %(policy)s because %(reason)s.')