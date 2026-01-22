import os
from taskflow import formatters
from taskflow.listeners import base
from taskflow import logging
from taskflow import states
from taskflow import task
from taskflow.types import failure
from taskflow.utils import misc
class LoggingListener(base.DumpingListener):
    """Listener that logs notifications it receives.

    It listens for task and flow notifications and writes those notifications
    to a provided logger, or logger of its module
    (``taskflow.listeners.logging``) if none is provided (and no class
    attribute is overridden). The log level can also be
    configured, ``logging.DEBUG`` is used by default when none is provided.
    """
    _LOGGER = None

    def __init__(self, engine, task_listen_for=base.DEFAULT_LISTEN_FOR, flow_listen_for=base.DEFAULT_LISTEN_FOR, retry_listen_for=base.DEFAULT_LISTEN_FOR, log=None, level=logging.DEBUG):
        super(LoggingListener, self).__init__(engine, task_listen_for=task_listen_for, flow_listen_for=flow_listen_for, retry_listen_for=retry_listen_for)
        self._logger = misc.pick_first_not_none(log, self._LOGGER, LOG)
        self._level = level

    def _dump(self, message, *args, **kwargs):
        self._logger.log(self._level, message, *args, **kwargs)