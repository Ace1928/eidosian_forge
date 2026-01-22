import os
import glob
from abc import abstractmethod
from unittest import mock
from typing import Any, Optional
import pytest
from queuelib.queue import (
from queuelib.tests import QueuelibTestCase
class LifoSQLiteQueueTest(LifoTestMixin, PersistentTestMixin, QueueTestMixin, QueuelibTestCase):

    def queue(self):
        return LifoSQLiteQueue(self.qpath)