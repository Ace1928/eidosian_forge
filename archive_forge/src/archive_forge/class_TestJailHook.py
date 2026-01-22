import threading
from breezy import errors, transport
from breezy.bzr.bzrdir import BzrDir
from breezy.bzr.smart import request
from breezy.tests import TestCase, TestCaseWithMemoryTransport
class TestJailHook(TestCaseWithMemoryTransport):

    def setUp(self):
        super().setUp()

        def clear_jail_info():
            request.jail_info.transports = None
        self.addCleanup(clear_jail_info)

    def test_jail_hook(self):
        request.jail_info.transports = None
        _pre_open_hook = request._pre_open_hook
        t = self.get_transport('foo')
        _pre_open_hook(t)
        request.jail_info.transports = [t]
        _pre_open_hook(t)
        _pre_open_hook(t.clone('child'))
        self.assertRaises(errors.JailBreak, _pre_open_hook, t.clone('..'))
        self.assertRaises(errors.JailBreak, _pre_open_hook, transport.get_transport_from_url('http://host/'))

    def test_open_bzrdir_in_non_main_thread(self):
        """Opening a bzrdir in a non-main thread should work ok.

        This makes sure that the globally-installed
        breezy.bzr.smart.request._pre_open_hook, which uses a threading.local(),
        works in a newly created thread.
        """
        bzrdir = self.make_controldir('.')
        transport = bzrdir.root_transport
        thread_result = []

        def t():
            BzrDir.open_from_transport(transport)
            thread_result.append('ok')
        thread = threading.Thread(target=t)
        thread.start()
        thread.join()
        self.assertEqual(['ok'], thread_result)