from breezy import osutils
from breezy.bzr.inventory import Inventory, InventoryFile
from breezy.bzr.tests.per_repository_vf import (
from breezy.repository import WriteGroup
from breezy.revision import NULL_REVISION, Revision
from breezy.tests import TestNotApplicable, multiply_scenarios
from breezy.tests.scenarios import load_tests_apply_scenarios
class BrokenRepoScenario:
    """Base class for defining scenarios for testing check and reconcile.

    A subclass needs to define the following methods:
        :populate_repository: a method to use to populate a repository with
            sample revisions, inventories and file versions.
        :all_versions_after_reconcile: all the versions in repository after
            reconcile.  run_test verifies that the text of each of these
            versions of the file is unchanged by the reconcile.
        :populated_parents: a list of (parents list, revision).  Each version
            of the file is verified to have the given parents before running
            the reconcile.  i.e. this is used to assert that the repo from the
            factory is what we expect.
        :corrected_parents: a list of (parents list, revision).  Each version
            of the file is verified to have the given parents after the
            reconcile.  i.e. this is used to assert that reconcile made the
            changes we expect it to make.

    A subclass may define the following optional method as well:
        :corrected_fulltexts: a list of file versions that should be stored as
            fulltexts (not deltas) after reconcile.  run_test will verify that
            this occurs.
    """

    def __init__(self, test_case):
        self.test_case = test_case

    def make_one_file_inventory(self, repo, revision, parents, inv_revision=None, root_revision=None, file_contents=None, make_file_version=True):
        return self.test_case.make_one_file_inventory(repo, revision, parents, inv_revision=inv_revision, root_revision=root_revision, file_contents=file_contents, make_file_version=make_file_version)

    def add_revision(self, repo, revision_id, inv, parent_ids):
        return self.test_case.add_revision(repo, revision_id, inv, parent_ids)

    def corrected_fulltexts(self):
        return []

    def repository_text_key_index(self):
        result = {}
        if self.versioned_root:
            result.update(self.versioned_repository_text_keys())
        result.update(self.repository_text_keys())
        return result