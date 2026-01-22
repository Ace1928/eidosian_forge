from __future__ import absolute_import, division, print_function
import stat
import os
import traceback
from ansible.module_utils.common import respawn
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils import distro
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import open_url
def _remove_repo(self):
    """Remove the required repository.

        Returns:
            True, if the repository has been removed, False otherwise.
        """
    self._read_all_repos()
    repo = self._get_copr_repo()
    if not repo:
        return False
    if not self.check_mode:
        try:
            os.remove(repo.repofile)
        except OSError as e:
            self.raise_exception(str(e))
    return True