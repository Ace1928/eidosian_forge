from oslo_utils import excutils
from neutron_lib._i18n import _
class PolicyInitError(NeutronException):
    """An error due to policy initialization failure.

    :param policy: The policy that failed to initialize.
    :param reason: Details on why the policy failed to initialize.
    """
    message = _('Failed to initialize policy %(policy)s because %(reason)s.')