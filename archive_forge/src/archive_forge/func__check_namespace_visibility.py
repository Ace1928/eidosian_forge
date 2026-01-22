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
def _check_namespace_visibility(context, namespace, namespace_name):
    if not _is_namespace_visible(context, namespace):
        LOG.debug('Forbidding request, metadata definition namespace=%s is not visible.', namespace_name)
        emsg = _('Forbidding request, metadata definition namespace=%s is not visible.') % namespace_name
        raise exception.MetadefForbidden(emsg)