from oslo_utils import excutils
from neutron_lib._i18n import _
class SubnetPoolDeleteError(BadRequest):
    message = _('Unable to delete subnet pool: %(reason)s.')