import copy
import logging
import os
import queue
import select
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from unittest import mock
import uuid
from oslo_utils import eventletutils
from oslo_utils import importutils
from string import Template
import testtools
import oslo_messaging
from oslo_messaging.tests import utils as test_utils
class _FailedResponder(_ListenerThread):

    def __init__(self, listener):
        super(_FailedResponder, self).__init__(listener, 1)

    def run(self):
        self.started.set()
        while not self._done.is_set():
            for in_msg in self.listener.poll(timeout=0.5):
                try:
                    raise RuntimeError('Oopsie!')
                except RuntimeError:
                    in_msg.reply(reply=None, failure=sys.exc_info())
                self._done.set()