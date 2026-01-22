from oslo_utils import reflection
from neutron_lib._i18n import _
from neutron_lib import exceptions
class NeutronObjectValidatorException(exceptions.NeutronException):
    message = _('Synthetic field(s) %(fields)s undefined, misspelled, or otherwise invalid')