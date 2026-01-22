import contextlib
import itertools
import logging
import os
import shutil
import socket
import sys
import tempfile
import threading
import time
from oslo_utils import timeutils
from oslo_utils import uuidutils
from zake import fake_client
from taskflow.conductors import backends as conductors
from taskflow import engines
from taskflow.jobs import backends as boards
from taskflow.patterns import linear_flow
from taskflow.persistence import backends as persistence
from taskflow.persistence import models
from taskflow import task
from taskflow.utils import threading_utils
class MakeTempDir(task.Task):
    default_provides = 'temp_dir'

    def execute(self):
        return tempfile.mkdtemp()

    def revert(self, *args, **kwargs):
        temp_dir = kwargs.get(task.REVERT_RESULT)
        if temp_dir:
            shutil.rmtree(temp_dir)