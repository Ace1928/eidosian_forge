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
class RangeConstrainedInteger(obj_fields.Integer):

    def __init__(self, start, end, **kwargs):
        try:
            self._start = int(start)
            self._end = int(end)
        except (TypeError, ValueError) as e:
            raise o_exc.NeutronRangeConstrainedIntegerInvalidLimit(start=start, end=end) from e
        super().__init__(**kwargs)

    def coerce(self, obj, attr, value):
        if not isinstance(value, int):
            msg = _('Field value %s is not an integer') % value
            raise ValueError(msg)
        if not self._start <= value <= self._end:
            msg = _('Field value %s is invalid') % value
            raise ValueError(msg)
        return super().coerce(obj, attr, value)