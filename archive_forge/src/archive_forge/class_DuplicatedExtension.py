from oslo_utils import excutils
from neutron_lib._i18n import _
class DuplicatedExtension(NeutronException):
    message = _('Found duplicate extension: %(alias)s.')