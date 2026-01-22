import itertools
import uuid
import netaddr
from oslo_serialization import jsonutils
from oslo_versionedobjects import fields as obj_fields
from neutron_lib._i18n import _
from neutron_lib import constants as lib_constants
from neutron_lib.db import constants as lib_db_const
from neutron_lib.objects import exceptions as o_exc
from neutron_lib.utils import net as net_utils
class DomainName(obj_fields.String):

    def coerce(self, obj, attr, value):
        if not isinstance(value, str):
            msg = _('Field value %s is not a string') % value
            raise ValueError(msg)
        if len(value) > lib_db_const.FQDN_FIELD_SIZE:
            msg = _('Domain name %s is too long') % value
            raise ValueError(msg)
        return super().coerce(obj, attr, value)