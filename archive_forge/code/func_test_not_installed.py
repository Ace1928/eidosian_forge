import os
import signal
import threading
import weakref
from breezy import tests, transport
from breezy.bzr.smart import client, medium, server, signals
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