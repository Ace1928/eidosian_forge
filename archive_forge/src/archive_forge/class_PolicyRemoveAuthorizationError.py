from neutron_lib._i18n import _
from neutron_lib import exceptions as e
class PolicyRemoveAuthorizationError(e.NotAuthorized):
    message = _('Failed to remove provided policy %(policy_id)s because you are not authorized.')