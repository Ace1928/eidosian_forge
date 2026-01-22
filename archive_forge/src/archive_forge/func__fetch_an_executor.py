import urllib
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import encodeutils
from oslo_utils import excutils
from stevedore import driver
from taskflow import engines
from taskflow.listeners import logging as llistener
import glance.async_
from glance.common import exception
from glance.common.scripts import utils as script_utils
from glance.i18n import _, _LE
@staticmethod
def _fetch_an_executor():
    if CONF.taskflow_executor.engine_mode != 'parallel':
        return None
    else:
        max_workers = CONF.taskflow_executor.max_workers
        threadpool_cls = glance.async_.get_threadpool_model()
        return threadpool_cls(max_workers).pool