import logging
import multiprocessing
import os
import signal
import socket
import time
import traceback
from unittest import mock
import eventlet
from eventlet import event
from oslotest import base as test_base
from oslo_service import service
from oslo_service.tests import base
from oslo_service.tests import eventlet_service
def _spawn_service(self, workers=1, service_maker=None, launcher_maker=None):
    self.workers = workers
    pid = os.fork()
    if pid == 0:
        os.setsid()
        status = 0
        try:
            serv = service_maker() if service_maker else ServiceWithTimer()
            if launcher_maker:
                launcher = launcher_maker()
                launcher.launch_service(serv, workers=workers)
            else:
                launcher = service.launch(self.conf, serv, workers=workers)
            status = launcher.wait()
        except SystemExit as exc:
            status = exc.code
        except BaseException:
            try:
                traceback.print_exc()
            except BaseException:
                print("Couldn't print traceback")
            status = 2
        os._exit(status or 0)
    return pid