import copy
import functools
import uuid
from oslo_config import cfg
from oslo_log import log as logging
from glance.common import exception
from glance.common import timeutils
from glance.common import utils
from glance.db import utils as db_utils
from glance.i18n import _, _LI, _LW
def _format_tag(values):
    dt = timeutils.utcnow()
    tag = {'id': _get_metadef_id(), 'namespace_id': None, 'name': None, 'created_at': dt, 'updated_at': dt}
    tag.update(values)
    return tag