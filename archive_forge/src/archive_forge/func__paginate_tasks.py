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
def _paginate_tasks(context, tasks, marker, limit, show_deleted):
    start = 0
    end = -1
    if marker is None:
        start = 0
    else:
        _task_get(context, marker, force_show_deleted=show_deleted)
        for i, task in enumerate(tasks):
            if task['id'] == marker:
                start = i + 1
                break
        else:
            if task:
                raise exception.TaskNotFound(task_id=task['id'])
            else:
                msg = _('Task does not exist')
                raise exception.NotFound(message=msg)
    end = start + limit if limit is not None else None
    return tasks[start:end]