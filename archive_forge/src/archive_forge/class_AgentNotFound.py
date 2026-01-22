from neutron_lib._i18n import _
from neutron_lib import exceptions
class AgentNotFound(exceptions.NotFound):
    message = _('Agent %(id)s could not be found.')