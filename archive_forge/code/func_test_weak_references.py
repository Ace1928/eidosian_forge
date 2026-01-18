import os
import signal
import threading
import weakref
from breezy import tests, transport
from breezy.bzr.smart import client, medium, server, signals
def test_weak_references(self):
    self.assertIsInstance(signals._on_sighup, weakref.WeakValueDictionary)
    calls = []

    def call_me():
        calls.append('called')
    signals.register_on_hangup('myid', call_me)
    del call_me
    signals._sighup_handler(SIGHUP, None)
    self.assertEqual([], calls)