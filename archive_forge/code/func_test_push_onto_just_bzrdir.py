import re
from breezy import (branch, controldir, directory_service, errors, osutils,
from breezy.bzr import bzrdir, knitrepo
from breezy.tests import http_server, scenarios, script, test_foreign
from breezy.transport import memory
def test_push_onto_just_bzrdir(self):
    """We don't handle when the target is just a bzrdir.

        Because you shouldn't be able to create *just* a bzrdir in the wild.
        """
    tree = self.create_simple_tree()
    a_controldir = self.make_controldir('dir')
    self.run_bzr_error(['At ../dir you have a valid .bzr control'], 'push ../dir', working_dir='tree')