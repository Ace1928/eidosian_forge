from breezy import config, tests
from breezy.tests import features
def test_remove_unknown_alias(self):
    out, err = self.run_bzr('alias --remove fooix', retcode=3)
    self.assertEqual('brz: ERROR: The alias "fooix" does not exist.\n', err)