from breezy import errors, gpg
from breezy import repository as _mod_repository
from breezy import revision as _mod_revision
from breezy import tests
from breezy.bzr import inventory, versionedfile, vf_repository
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.matchers import MatchesAncestry
from breezy.tests.scenarios import load_tests_apply_scenarios
def assertFormatAttribute(self, attribute, allowed_values):
    """Assert that the format has an attribute 'attribute'."""
    repo = self.make_repository('repo')
    self.assertSubset([getattr(repo._format, attribute)], allowed_values)