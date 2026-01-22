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
class ServiceTest(test_base.BaseTestCase):

    def test_graceful_stop(self):
        self.assertIsNone(exercise_graceful_test_service(1, 2, True))

    def test_ungraceful_stop(self):
        self.assertEqual('Timeout!', exercise_graceful_test_service(1, 2, False))