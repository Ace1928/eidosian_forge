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
def _filter_tasks(tasks, filters, context, admin_as_user=False):
    filtered_tasks = []
    for task in tasks:
        has_ownership = context.owner and task['owner'] == context.owner
        can_see = has_ownership or (context.is_admin and (not admin_as_user))
        if not can_see:
            continue
        add = True
        for k, value in filters.items():
            add = task[k] == value and task['deleted'] is False
            if not add:
                break
        if add:
            filtered_tasks.append(task)
    return filtered_tasks