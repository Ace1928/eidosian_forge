import sys
import pytest
class TestWithoutSSL(object):

    @classmethod
    def setup_class(cls):
        sys.modules.pop('ssl', None)
        sys.modules.pop('_ssl', None)
        module_stash.stash()
        sys.meta_path.insert(0, ssl_blocker)

    def teardown_class(cls):
        sys.meta_path.remove(ssl_blocker)
        module_stash.pop()