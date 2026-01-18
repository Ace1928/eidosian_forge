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
def test_setup_signal_interruption_select_poll(self):
    service.SignalHandler.__class__._instances.clear()
    signal_handler = service.SignalHandler()
    self.addCleanup(service.SignalHandler.__class__._instances.clear)
    self.assertTrue(signal_handler._SignalHandler__force_interrupt_on_signal)