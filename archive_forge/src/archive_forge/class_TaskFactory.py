from collections import abc
import datetime
import uuid
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import importutils
from glance.common import exception
from glance.common import timeutils
from glance.i18n import _, _LE, _LI, _LW
class TaskFactory(object):

    def new_task(self, task_type, owner, image_id, user_id, request_id, task_input=None, **kwargs):
        task_id = str(uuid.uuid4())
        status = 'pending'
        expires_at = None
        created_at = timeutils.utcnow()
        updated_at = created_at
        return Task(task_id, task_type, status, owner, image_id, user_id, request_id, expires_at, created_at, updated_at, task_input, kwargs.get('result'), kwargs.get('message'))