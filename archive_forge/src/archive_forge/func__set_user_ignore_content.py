from breezy import bedding, ignores, tests
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def _set_user_ignore_content(self, ignores):
    """Create user ignore file and set its content to ignores."""
    bedding.ensure_config_dir_exists()
    user_ignore_file = bedding.user_ignore_config_path()
    with open(user_ignore_file, 'wb') as f:
        f.write(ignores)