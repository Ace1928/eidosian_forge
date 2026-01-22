from __future__ import absolute_import, division, print_function
import os
import os.path
import re
import shutil
import subprocess
import tempfile
import time
import shlex
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.parsing.convert_bool import BOOLEANS_FALSE
from ansible.module_utils.common.text.converters import to_text, to_bytes
def _rsync_data(self, container_path, temp_dir):
    """Sync the container directory to the temp directory.

        :param container_path: path to the container container
        :type container_path: ``str``
        :param temp_dir: path to the temporary local working directory
        :type temp_dir: ``str``
        """
    fs_paths = container_path.split(':')
    if 'overlayfs' in fs_paths:
        fs_paths.pop(fs_paths.index('overlayfs'))
    for fs_path in fs_paths:
        fs_path = os.path.dirname(fs_path)
        build_command = [self.module.get_bin_path('rsync', True), '-aHAX', fs_path, temp_dir]
        rc, stdout, err = self.module.run_command(build_command)
        if rc != 0:
            self.failure(err=err, rc=rc, msg='failed to perform archive', command=' '.join(build_command))