import collections
import logging
from unittest import mock
import fixtures
from oslotest import base
from testtools import compat
from testtools import matchers
from testtools import testcase
from taskflow import exceptions
from taskflow.tests import fixtures as taskflow_fixtures
from taskflow.tests import utils
from taskflow.utils import misc
class CapturingLoggingHandler(logging.Handler):
    """A handler that saves record contents for post-test analysis."""

    def __init__(self, level=logging.DEBUG):
        logging.Handler.__init__(self, level=level)
        self._records = []

    @property
    def counts(self):
        """Returns a dictionary with the number of records at each level."""
        self.acquire()
        try:
            captured = collections.defaultdict(int)
            for r in self._records:
                captured[r.levelno] += 1
            return captured
        finally:
            self.release()

    @property
    def messages(self):
        """Returns a dictionary with list of record messages at each level."""
        self.acquire()
        try:
            captured = collections.defaultdict(list)
            for r in self._records:
                captured[r.levelno].append(r.getMessage())
            return captured
        finally:
            self.release()

    @property
    def exc_infos(self):
        """Returns a list of all the record exc_info tuples captured."""
        self.acquire()
        try:
            captured = []
            for r in self._records:
                if r.exc_info:
                    captured.append(r.exc_info)
            return captured
        finally:
            self.release()

    def emit(self, record):
        self.acquire()
        try:
            self._records.append(record)
        finally:
            self.release()

    def reset(self):
        """Resets *all* internally captured state."""
        self.acquire()
        try:
            self._records = []
        finally:
            self.release()

    def close(self):
        logging.Handler.close(self)
        self.reset()