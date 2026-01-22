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
class ServiceManagerTestCase(test_base.BaseTestCase):
    """Test cases for Services."""

    def test_override_manager_method(self):
        serv = ExtendedService()
        serv.start()
        self.assertEqual('service', serv.test_method())