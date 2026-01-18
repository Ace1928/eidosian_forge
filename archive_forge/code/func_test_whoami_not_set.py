from breezy import branch, config, errors, tests
from ..test_bedding import override_whoami
def test_whoami_not_set(self):
    """Ensure whoami error if username is not set and not inferred.
        """
    override_whoami(self)
    out, err = self.run_bzr(['whoami'], 3)
    self.assertContainsRe(err, 'Unable to determine your name')