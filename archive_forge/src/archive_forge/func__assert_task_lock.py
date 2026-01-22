import copy
import functools
import json
import os
import urllib.request
import glance_store as store_api
from glance_store import backend
from glance_store import exceptions as store_exceptions
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import encodeutils
from oslo_utils import excutils
from oslo_utils import timeutils
from oslo_utils import units
import taskflow
from taskflow.patterns import linear_flow as lf
from taskflow import retry
from taskflow import task
from glance.api import common as api_common
import glance.async_.flows._internal_plugins as internal_plugins
import glance.async_.flows.plugins as import_plugins
from glance.async_ import utils
from glance.common import exception
from glance.common.scripts.image_import import main as image_import
from glance.common.scripts import utils as script_utils
from glance.common import store_utils
from glance.i18n import _, _LE, _LI
from glance.quota import keystone as ks_quota
def _assert_task_lock(self, image):
    task_lock = image.extra_properties.get('os_glance_import_task')
    if task_lock != self._task_id:
        LOG.error('Image %(image)s import task %(task)s attempted to take action on image, but other task %(other)s holds the lock; Aborting.', {'image': self._image_id, 'task': self._task_id, 'other': task_lock})
        raise exception.TaskAbortedError()