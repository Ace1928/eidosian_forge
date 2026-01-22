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
def _pop_task_info_values(values):
    task_info_values = {}
    for k, v in list(values.items()):
        if k in ['input', 'result', 'message']:
            values.pop(k)
            task_info_values[k] = v
    return task_info_values