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
def _task_info_get(task_id):
    """Get Task Info for Task with given task ID"""
    global DATA
    try:
        task_info = DATA['task_info'][task_id]
    except KeyError:
        msg = _LW('Could not find task info %s') % task_id
        LOG.warning(msg)
        raise exception.TaskNotFound(task_id=task_id)
    return task_info