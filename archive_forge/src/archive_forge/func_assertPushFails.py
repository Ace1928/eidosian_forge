import re
from breezy import (branch, controldir, directory_service, errors, osutils,
from breezy.bzr import bzrdir, knitrepo
from breezy.tests import http_server, scenarios, script, test_foreign
from breezy.transport import memory
def assertPushFails(self, args):
    out, err = self.run_bzr_error(self._default_errors, self._default_command + args, working_dir=self._default_wd, retcode=3)
    self.assertContainsRe(err, self._default_additional_error)