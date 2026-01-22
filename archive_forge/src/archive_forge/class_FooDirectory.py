import re
from breezy import (branch, controldir, directory_service, errors, osutils,
from breezy.bzr import bzrdir, knitrepo
from breezy.tests import http_server, scenarios, script, test_foreign
from breezy.transport import memory
class FooDirectory:

    def look_up(self, name, url, purpose=None):
        if url == 'foo:':
            return trunk_tree.branch.base
        return url