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
class ServiceWithTimer(service.Service):

    def __init__(self, ready_event=None):
        super(ServiceWithTimer, self).__init__()
        self.ready_event = ready_event

    def start(self):
        super(ServiceWithTimer, self).start()
        self.timer_fired = 0
        self.tg.add_timer(1, self.timer_expired)

    def wait(self):
        if self.ready_event:
            self.ready_event.set()
        super(ServiceWithTimer, self).wait()

    def timer_expired(self):
        self.timer_fired = self.timer_fired + 1