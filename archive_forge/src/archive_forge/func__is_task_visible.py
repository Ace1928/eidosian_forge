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
def _is_task_visible(context, task):
    """Return True if the task is visible in this context."""
    if context.is_admin:
        return True
    if task['owner'] is None:
        return True
    if context.owner is not None:
        if context.owner == task['owner']:
            return True
    return False