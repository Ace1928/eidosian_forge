from __future__ import (absolute_import, division, print_function)
import base64
import json
import os
import os.path
import shutil
import tempfile
import traceback
import zipfile
from ansible import constants as C
from ansible.errors import AnsibleError, AnsibleFileNotFound
from ansible.module_utils._text import to_bytes, to_native, to_text
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.plugins.action import ActionBase
from ansible.utils.hashing import checksum
def _copy_zip_file(self, dest, files, directories, task_vars, tmp, backup):
    if self._play_context.check_mode:
        module_return = dict(changed=True)
        return module_return
    try:
        zip_file = self._create_zip_tempfile(files, directories)
    except Exception as e:
        module_return = dict(changed=False, failed=True, msg='failed to create tmp zip file: %s' % to_text(e), exception=traceback.format_exc())
        return module_return
    zip_path = self._loader.get_real_file(zip_file)
    tmp_src = self._connection._shell.join_path(tmp, 'source.zip')
    self._transfer_file(zip_path, tmp_src)
    copy_args = self._task.args.copy()
    copy_args.update(dict(src=tmp_src, dest=dest, _copy_mode='explode', backup=backup))
    copy_args.pop('content', None)
    module_return = self._execute_module(module_name='ansible.windows.win_copy', module_args=copy_args, task_vars=task_vars)
    shutil.rmtree(os.path.dirname(zip_path))
    return module_return