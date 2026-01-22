from oslo_db.sqlalchemy import utils as sa_utils
from sqlalchemy.orm import lazyload
from sqlalchemy import sql, or_, and_
from neutron_lib._i18n import _
from neutron_lib.api import attributes
from neutron_lib import constants
from neutron_lib.db import utils as db_utils
from neutron_lib import exceptions as n_exc
from neutron_lib.objects import utils as obj_utils
from neutron_lib.utils import helpers
def _unique_keys(model):
    uk_sets = sa_utils.get_unique_keys(model)
    return uk_sets[0] if uk_sets else []