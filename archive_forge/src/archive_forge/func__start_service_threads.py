import eventlet
import __original_module_threading as orig_threading
import threading  # noqa
import sys
from oslo_config import cfg
import oslo_i18n as i18n
from oslo_log import log as logging
from oslo_service import systemd
from heat.cmd import api
from heat.cmd import api_cfn
from heat.cmd import engine
from heat.common import config
from heat.common import messaging
from heat import version
def _start_service_threads(services):
    threads = []
    for option in services:
        launch_func = LAUNCH_SERVICES[option][0]
        kwargs = LAUNCH_SERVICES[option][1]
        threads.append(eventlet.spawn(launch_func, **kwargs))
    return threads