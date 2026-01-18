import threading
import time
from eventlet.green import threading as green_threading
import testscenarios
from testtools import testcase
import futurist
from futurist import rejection
from futurist.tests import base
def wait_until_set(check_delay):
    ev_thread_started.set()
    while not ev.is_set():
        ev.wait(check_delay)