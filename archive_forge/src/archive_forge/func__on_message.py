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
def _on_message(message, handle, link):
    if self._nack_count:
        self._nack_count -= 1
        nack_method(link, handle)
    else:
        self._broker.forward_message(message, handle, link)