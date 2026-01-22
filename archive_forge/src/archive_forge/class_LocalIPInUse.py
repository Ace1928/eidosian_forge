from neutron_lib._i18n import _
from neutron_lib import exceptions
class LocalIPInUse(exceptions.InUse):
    message = _('Local IP %(id)s is still associated with one or more ports.')