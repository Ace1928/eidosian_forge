from oslo_utils import reflection
from neutron_lib._i18n import _
from neutron_lib import exceptions
class NeutronDbObjectNotFoundByModel(exceptions.NotFound):
    message = _('NeutronDbObject not found by model %(model)s.')