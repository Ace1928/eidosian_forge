import contextlib
import io
import logging
import sys
import threading
import time
import unittest
from traits.api import HasTraits, Str, Int, Float, Any, Event
from traits.api import push_exception_handler, pop_exception_handler
class TestListeners(unittest.TestCase):

    def test_listeners(self):
        global events
        ge = GenerateEvents()
        le = ListenEvents()
        ge.trait_set(name='Joe', age=22, weight=152.0)
        ge.add_trait_listener(le)
        events = {}
        ge.trait_set(name='Mike', age=34, weight=178.0)
        self.assertEqual(events, {'_age_changed': ('age', 22, 34), '_weight_changed': ('weight', 152.0, 178.0), '_name_changed': ('name', 'Joe', 'Mike')})
        ge.add_trait_listener(le, 'alt')
        events = {}
        ge.trait_set(name='Gertrude', age=39, weight=108.0)
        self.assertEqual(events, {'_age_changed': ('age', 34, 39), '_name_changed': ('name', 'Mike', 'Gertrude'), '_weight_changed': ('weight', 178.0, 108.0), 'alt_name_changed': ('name', 'Mike', 'Gertrude'), 'alt_weight_changed': ('weight', 178.0, 108.0)})
        ge.remove_trait_listener(le)
        events = {}
        ge.trait_set(name='Sally', age=46, weight=118.0)
        self.assertEqual(events, {'alt_name_changed': ('name', 'Gertrude', 'Sally'), 'alt_weight_changed': ('weight', 108.0, 118.0)})
        ge.remove_trait_listener(le, 'alt')
        events = {}
        ge.trait_set(name='Ralph', age=29, weight=198.0)
        self.assertEqual(events, {})

    def test_trait_exception_handler_can_access_exception(self):
        """ Tests if trait exception handlers can access the traceback of the
        exception.
        """
        from traits import trait_notifiers

        def _handle_exception(obj, name, old, new):
            self.assertIsNotNone(sys.exc_info()[0])
        ge = GenerateFailingEvents()
        try:
            trait_notifiers.push_exception_handler(_handle_exception, reraise_exceptions=False, main=True)
            ge.trait_set(name='John Cleese')
        finally:
            trait_notifiers.pop_exception_handler()

    def test_exceptions_logged(self):
        ge = GenerateFailingEvents()
        traits_logger = logging.getLogger('traits')
        with self.assertLogs(logger=traits_logger, level=logging.ERROR) as log_watcher:
            ge.name = 'Terry Jones'
        self.assertEqual(len(log_watcher.records), 1)
        log_record = log_watcher.records[0]
        self.assertIn('Exception occurred in traits notification handler', log_record.message)
        _, exc_value, exc_traceback = log_record.exc_info
        self.assertIsInstance(exc_value, RuntimeError)
        self.assertIsNotNone(exc_traceback)