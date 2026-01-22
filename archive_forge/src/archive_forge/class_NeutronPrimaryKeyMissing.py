from oslo_utils import reflection
from neutron_lib._i18n import _
from neutron_lib import exceptions
class NeutronPrimaryKeyMissing(exceptions.BadRequest):
    message = _('For class %(object_type)s missing primary keys: %(missing_keys)s')

    def __init__(self, object_class, missing_keys):
        super().__init__(object_type=reflection.get_class_name(object_class, fully_qualified=False), missing_keys=missing_keys)