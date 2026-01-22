from neutron_lib._i18n import _
from neutron_lib import exceptions
class IkeValidationFailure(exceptions.BadRequest):
    message = _("IKE does not support %(resource)s attribute %(key)s with value '%(value)s'")