from neutron_lib._i18n import _
from neutron_lib import exceptions as qexception
class TapFlowNotFound(qexception.NotFound):
    message = _('Tap Flow  %(flow_id)s does not exist')