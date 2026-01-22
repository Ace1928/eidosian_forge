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
class IntegerEnum(obj_fields.Integer):

    def __init__(self, valid_values=None, **kwargs):
        if not valid_values:
            msg = _('No possible values specified')
            raise ValueError(msg)
        for value in valid_values:
            if not isinstance(value, int):
                msg = _('Possible value %s is not an integer') % value
                raise ValueError(msg)
        self._valid_values = valid_values
        super().__init__(**kwargs)

    def coerce(self, obj, attr, value):
        if not isinstance(value, int):
            msg = _('Field value %s is not an integer') % value
            raise ValueError(msg)
        if value not in self._valid_values:
            msg = _('Field value %(value)s is not in the list of valid values: %(values)s') % {'value': value, 'values': self._valid_values}
            raise ValueError(msg)
        return super().coerce(obj, attr, value)