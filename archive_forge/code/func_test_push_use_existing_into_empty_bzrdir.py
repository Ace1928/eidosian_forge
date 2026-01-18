import re
from breezy import (branch, controldir, directory_service, errors, osutils,
from breezy.bzr import bzrdir, knitrepo
from breezy.tests import http_server, scenarios, script, test_foreign
from breezy.transport import memory
def test_push_use_existing_into_empty_bzrdir(self):
    """'brz push --use-existing-dir' into a dir with an empty .bzr dir
        fails.
        """
    tree = self.create_simple_tree()
    self.build_tree(['target/', 'target/.bzr/'])
    self.run_bzr_error(['Target directory ../target already contains a .bzr directory, but it is not valid.'], 'push ../target --use-existing-dir', working_dir='tree')