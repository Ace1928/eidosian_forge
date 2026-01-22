import socketserver
from .. import errors, tests
from ..bzr.tests import test_read_bundle
from ..directory_service import directories
from ..mergeable import read_mergeable_from_url
from . import test_server
class TestReadMergeableFromUrl(tests.TestCaseWithTransport):

    def test_read_mergeable_skips_local(self):
        """A local bundle named like the URL should not be read.
        """
        out, wt = test_read_bundle.create_bundle_file(self)

        class FooService:
            """A directory service that always returns source"""

            def look_up(self, name, url):
                return 'source'
        directories.register('foo:', FooService, 'Testing directory service')
        self.addCleanup(directories.remove, 'foo:')
        self.build_tree_contents([('./foo:bar', out.getvalue())])
        self.assertRaises(errors.NotABundle, read_mergeable_from_url, 'foo:bar')

    def test_infinite_redirects_are_not_a_bundle(self):
        """If a URL causes TooManyRedirections then NotABundle is raised.
        """
        from .blackbox.test_push import RedirectingMemoryServer
        server = RedirectingMemoryServer()
        self.start_server(server)
        url = server.get_url() + 'infinite-loop'
        self.assertRaises(errors.NotABundle, read_mergeable_from_url, url)

    def test_smart_server_connection_reset(self):
        """If a smart server connection fails during the attempt to read a
        bundle, then the ConnectionReset error should be propagated.
        """
        sock_server = DisconnectingServer()
        self.start_server(sock_server)
        url = sock_server.get_url()
        self.assertRaises(errors.ConnectionReset, read_mergeable_from_url, url)