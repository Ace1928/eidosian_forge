from neutron_lib._i18n import _
from neutron_lib import exceptions
class WrongEndpointGroupType(exceptions.BadRequest):
    message = _("Endpoint group %(which)s type is '%(group_type)s' and should be '%(expected)s'")