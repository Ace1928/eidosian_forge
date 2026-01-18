import os
from breezy import tests
from breezy.tests import script
def test_unknnown_file(self):
    self.run_bzr(['test-script', 'I-do-not-exist'], retcode=3)