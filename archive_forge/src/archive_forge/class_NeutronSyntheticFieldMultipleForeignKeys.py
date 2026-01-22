from oslo_utils import reflection
from neutron_lib._i18n import _
from neutron_lib import exceptions
class NeutronSyntheticFieldMultipleForeignKeys(exceptions.NeutronException):
    message = _("Synthetic field %(field)s shouldn't have more than one foreign key")