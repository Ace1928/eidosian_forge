import threading
from unittest import mock
from oslo_concurrency import processutils as putils
from oslo_context import context as context_utils
from os_brick import executor as brick_executor
from os_brick.privileged import rootwrap
from os_brick.tests import base
def _run_threads(self, threads):
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()