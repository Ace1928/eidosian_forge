import socketserver
from .. import errors, tests
from ..bzr.tests import test_read_bundle
from ..directory_service import directories
from ..mergeable import read_mergeable_from_url
from . import test_server
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