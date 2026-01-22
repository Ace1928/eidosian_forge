import os
import signal
import threading
import weakref
from breezy import tests, transport
from breezy.bzr.smart import client, medium, server, signals
class TestSignalHandlers(tests.TestCase):

    def setUp(self):
        super().setUp()
        orig = signals._setup_on_hangup_dict()
        self.assertIs(None, orig)

        def cleanup():
            signals._on_sighup = None
        self.addCleanup(cleanup)

    def test_registered_callback_gets_called(self):
        calls = []

        def call_me():
            calls.append('called')
        signals.register_on_hangup('myid', call_me)
        signals._sighup_handler(SIGHUP, None)
        self.assertEqual(['called'], calls)
        signals.unregister_on_hangup('myid')

    def test_unregister_not_present(self):
        signals.unregister_on_hangup('no-such-id')
        log = self.get_log()
        self.assertContainsRe(log, 'Error occurred during unregister_on_hangup:')
        self.assertContainsRe(log, '(?s)Traceback.*KeyError')

    def test_failing_callback(self):
        calls = []

        def call_me():
            calls.append('called')

        def fail_me():
            raise RuntimeError('something bad happened')
        signals.register_on_hangup('myid', call_me)
        signals.register_on_hangup('otherid', fail_me)
        signals._sighup_handler(SIGHUP, None)
        signals.unregister_on_hangup('myid')
        signals.unregister_on_hangup('otherid')
        log = self.get_log()
        self.assertContainsRe(log, '(?s)Traceback.*RuntimeError')
        self.assertEqual(['called'], calls)

    def test_unregister_during_call(self):
        calls = []

        def call_me_and_unregister():
            signals.unregister_on_hangup('myid')
            calls.append('called_and_unregistered')

        def call_me():
            calls.append('called')
        signals.register_on_hangup('myid', call_me_and_unregister)
        signals.register_on_hangup('other', call_me)
        signals._sighup_handler(SIGHUP, None)

    def test_keyboard_interrupt_propagated(self):

        def call_me_and_raise():
            raise KeyboardInterrupt()
        signals.register_on_hangup('myid', call_me_and_raise)
        self.assertRaises(KeyboardInterrupt, signals._sighup_handler, SIGHUP, None)
        signals.unregister_on_hangup('myid')

    def test_weak_references(self):
        self.assertIsInstance(signals._on_sighup, weakref.WeakValueDictionary)
        calls = []

        def call_me():
            calls.append('called')
        signals.register_on_hangup('myid', call_me)
        del call_me
        signals._sighup_handler(SIGHUP, None)
        self.assertEqual([], calls)

    def test_not_installed(self):
        signals._on_sighup = None
        calls = []

        def call_me():
            calls.append('called')
        signals.register_on_hangup('myid', calls)
        signals._sighup_handler(SIGHUP, None)
        signals.unregister_on_hangup('myid')
        log = self.get_log()
        self.assertEqual('', log)

    def test_install_sighup_handler(self):
        signals._on_sighup = None
        orig = signals.install_sighup_handler()
        if getattr(signal, 'SIGHUP', None) is not None:
            cur = signal.getsignal(SIGHUP)
            self.assertEqual(signals._sighup_handler, cur)
        self.assertIsNot(None, signals._on_sighup)
        signals.restore_sighup_handler(orig)
        self.assertIs(None, signals._on_sighup)