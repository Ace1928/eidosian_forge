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
def _create_tar(self, source_dir):
    """Create an archive of a given ``source_dir`` to ``output_path``.

        :param source_dir:  Path to the directory to be archived.
        :type source_dir: ``str``
        """
    old_umask = os.umask(int('0077', 8))
    archive_path = self.module.params['archive_path']
    if not os.path.isdir(archive_path):
        os.makedirs(archive_path)
    archive_compression = self.module.params['archive_compression']
    compression_type = LXC_COMPRESSION_MAP[archive_compression]
    archive_name = '%s.%s' % (os.path.join(archive_path, self.container_name), compression_type['extension'])
    build_command = [self.module.get_bin_path('tar', True), '--directory=%s' % os.path.realpath(source_dir), compression_type['argument'], archive_name, '.']
    rc, stdout, err = self.module.run_command(build_command)
    os.umask(old_umask)
    if rc != 0:
        self.failure(err=err, rc=rc, msg='failed to create tar archive', command=' '.join(build_command))
    return archive_name