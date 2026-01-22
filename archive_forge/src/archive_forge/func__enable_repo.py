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
def _enable_repo(self, repo_filename_path, repo_content=None):
    """Write information to a repo file.

        Args:
            repo_filename_path: Path to repository.
            repo_content: Repository information from the host.

        Returns:
            True, if the information in the repo file matches that stored on the host,
            False otherwise.
        """
    if not repo_content:
        repo_content = self._download_repo_info()
    if self._compare_repo_content(repo_filename_path, repo_content):
        return False
    if not self.check_mode:
        with open(repo_filename_path, 'w+') as file:
            file.write(repo_content)
        os.chmod(repo_filename_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
    return True