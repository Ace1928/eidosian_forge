import contextlib
import logging
import threading
import time
from oslo_serialization import jsonutils
from oslo_utils import reflection
from zake import fake_client
import taskflow.engines
from taskflow import exceptions as exc
from taskflow.jobs import backends as jobs
from taskflow.listeners import claims
from taskflow.listeners import logging as logging_listeners
from taskflow.listeners import timing
from taskflow.patterns import linear_flow as lf
from taskflow.persistence.backends import impl_memory
from taskflow import states
from taskflow import task
from taskflow import test
from taskflow.test import mock
from taskflow.tests import utils as test_utils
from taskflow.utils import misc
from taskflow.utils import persistence_utils
class TestLoggingListeners(test.TestCase, EngineMakerMixin):

    def _make_logger(self, level=logging.DEBUG):
        log = logging.getLogger(reflection.get_callable_name(self._get_test_method()))
        log.propagate = False
        for handler in reversed(log.handlers):
            log.removeHandler(handler)
        handler = test.CapturingLoggingHandler(level=level)
        log.addHandler(handler)
        log.setLevel(level)
        self.addCleanup(handler.reset)
        self.addCleanup(log.removeHandler, handler)
        return (log, handler)

    def test_basic(self):
        flow = lf.Flow('test')
        flow.add(test_utils.TaskNoRequiresNoReturns('test-1'))
        e = self._make_engine(flow)
        log, handler = self._make_logger()
        with logging_listeners.LoggingListener(e, log=log):
            e.run()
        self.assertGreater(0, handler.counts[logging.DEBUG])
        for levelno in _LOG_LEVELS - set([logging.DEBUG]):
            self.assertEqual(0, handler.counts[levelno])
        self.assertEqual([], handler.exc_infos)

    def test_basic_customized(self):
        flow = lf.Flow('test')
        flow.add(test_utils.TaskNoRequiresNoReturns('test-1'))
        e = self._make_engine(flow)
        log, handler = self._make_logger()
        listener = logging_listeners.LoggingListener(e, log=log, level=logging.INFO)
        with listener:
            e.run()
        self.assertGreater(0, handler.counts[logging.INFO])
        for levelno in _LOG_LEVELS - set([logging.INFO]):
            self.assertEqual(0, handler.counts[levelno])
        self.assertEqual([], handler.exc_infos)

    def test_basic_failure(self):
        flow = lf.Flow('test')
        flow.add(test_utils.TaskWithFailure('test-1'))
        e = self._make_engine(flow)
        log, handler = self._make_logger()
        with logging_listeners.LoggingListener(e, log=log):
            self.assertRaises(RuntimeError, e.run)
        self.assertGreater(0, handler.counts[logging.DEBUG])
        for levelno in _LOG_LEVELS - set([logging.DEBUG]):
            self.assertEqual(0, handler.counts[levelno])
        self.assertEqual(1, len(handler.exc_infos))

    def test_dynamic(self):
        flow = lf.Flow('test')
        flow.add(test_utils.TaskNoRequiresNoReturns('test-1'))
        e = self._make_engine(flow)
        log, handler = self._make_logger()
        with logging_listeners.DynamicLoggingListener(e, log=log):
            e.run()
        self.assertGreater(0, handler.counts[logging.DEBUG])
        for levelno in _LOG_LEVELS - set([logging.DEBUG]):
            self.assertEqual(0, handler.counts[levelno])
        self.assertEqual([], handler.exc_infos)

    def test_dynamic_failure(self):
        flow = lf.Flow('test')
        flow.add(test_utils.TaskWithFailure('test-1'))
        e = self._make_engine(flow)
        log, handler = self._make_logger()
        with logging_listeners.DynamicLoggingListener(e, log=log):
            self.assertRaises(RuntimeError, e.run)
        self.assertGreater(0, handler.counts[logging.WARNING])
        self.assertGreater(0, handler.counts[logging.DEBUG])
        self.assertEqual(1, len(handler.exc_infos))
        for levelno in _LOG_LEVELS - set([logging.DEBUG, logging.WARNING]):
            self.assertEqual(0, handler.counts[levelno])

    def test_dynamic_failure_customized_level(self):
        flow = lf.Flow('test')
        flow.add(test_utils.TaskWithFailure('test-1'))
        e = self._make_engine(flow)
        log, handler = self._make_logger()
        listener = logging_listeners.DynamicLoggingListener(e, log=log, failure_level=logging.ERROR)
        with listener:
            self.assertRaises(RuntimeError, e.run)
        self.assertGreater(0, handler.counts[logging.ERROR])
        self.assertGreater(0, handler.counts[logging.DEBUG])
        self.assertEqual(1, len(handler.exc_infos))
        for levelno in _LOG_LEVELS - set([logging.DEBUG, logging.ERROR]):
            self.assertEqual(0, handler.counts[levelno])