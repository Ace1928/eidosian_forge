from typing import List
from .. import urlutils
from ..branch import Branch
from ..bzr import BzrProber
from ..bzr.branch import BranchReferenceFormat
from ..controldir import ControlDir, ControlDirFormat
from ..errors import NotBranchError, RedirectRequested
from ..transport import (Transport, chroot, get_transport, register_transport,
from ..url_policy_open import (BadUrl, BranchLoopError, BranchOpener,
from . import TestCase, TestCaseWithTransport
class TestOpenOnlyScheme(TestCaseWithTransport):
    """Tests for `open_only_scheme`."""

    def setUp(self):
        super().setUp()
        BranchOpener.install_hook()

    def test_hook_does_not_interfere(self):
        self.make_branch('stacked')
        self.make_branch('stacked-on')
        Branch.open('stacked').set_stacked_on_url('../stacked-on')
        Branch.open('stacked')

    def get_chrooted_scheme(self, relpath):
        """Create a server that is chrooted to `relpath`.

        :return: ``(scheme, get_url)`` where ``scheme`` is the scheme of the
            chroot server and ``get_url`` returns URLs on said server.
        """
        transport = self.get_transport(relpath)
        chroot_server = chroot.ChrootServer(transport)
        chroot_server.start_server()
        self.addCleanup(chroot_server.stop_server)

        def get_url(relpath):
            return chroot_server.get_url() + relpath
        return (urlutils.URL.from_string(chroot_server.get_url()).scheme, get_url)

    def test_stacked_within_scheme(self):
        self.get_transport().mkdir('inside')
        self.make_branch('inside/stacked')
        self.make_branch('inside/stacked-on')
        scheme, get_chrooted_url = self.get_chrooted_scheme('inside')
        Branch.open(get_chrooted_url('stacked')).set_stacked_on_url(get_chrooted_url('stacked-on'))
        open_only_scheme(scheme, get_chrooted_url('stacked'))

    def test_stacked_outside_scheme(self):
        self.get_transport().mkdir('inside')
        self.get_transport().mkdir('outside')
        self.make_branch('inside/stacked')
        self.make_branch('outside/stacked-on')
        scheme, get_chrooted_url = self.get_chrooted_scheme('inside')
        Branch.open(get_chrooted_url('stacked')).set_stacked_on_url(self.get_url('outside/stacked-on'))
        self.assertRaises(BadUrl, open_only_scheme, scheme, get_chrooted_url('stacked'))