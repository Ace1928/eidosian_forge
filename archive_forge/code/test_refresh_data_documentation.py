from breezy import errors, repository
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.scenarios import load_tests_apply_scenarios
Create a new revision (revid 'new-rev') and fetch it into a
        concurrent instance of repo.
        