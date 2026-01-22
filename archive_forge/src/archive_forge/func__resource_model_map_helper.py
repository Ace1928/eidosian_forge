from oslo_utils import timeutils
import sqlalchemy as sa
from sqlalchemy import event  # noqa
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.ext import declarative
from sqlalchemy.orm import attributes
from sqlalchemy.orm import session as se
from neutron_lib._i18n import _
from neutron_lib.db import constants as db_const
from neutron_lib.db import model_base
from neutron_lib.db import sqlalchemytypes
def _resource_model_map_helper(rs_map, resource, subclass):
    if resource in rs_map:
        raise RuntimeError(_('Model %(sub)s tried to register for API resource %(res)s which conflicts with model %(other)s.') % {'sub': subclass, 'other': rs_map[resource], 'res': resource})
    rs_map[resource] = subclass