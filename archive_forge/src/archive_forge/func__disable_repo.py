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
def _disable_repo(self, repo_filename_path):
    """Disable the repository.

        Args:
            repo_filename_path: Path to repository.

        Returns:
            False, if the repository is already disabled on the system,
            True otherwise.
        """
    self._read_all_repos()
    repo = self._get_copr_repo()
    if repo is None:
        if self.check_mode:
            return True
        self._enable_repo(repo_filename_path)
        self._read_all_repos('copr:{0}:{1}:{2}'.format(self.host, self.user, self.project))
        repo = self._get_copr_repo()
    for repo_id in repo.cfg.sections():
        repo_content_api = self._download_repo_info()
        with open(repo_filename_path, 'r') as file:
            repo_content_file = file.read()
        if repo_content_file != repo_content_api:
            if not self.resolve_differences(repo_content_file, repo_content_api, repo_filename_path):
                return False
        if not self.check_mode:
            self.base.conf.write_raw_configfile(repo.repofile, repo_id, self.base.conf.substitutions, {'enabled': '0'})
    return True