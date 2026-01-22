from __future__ import (absolute_import, division, print_function)
import os.path
from ansible import constants as C
from ansible.module_utils.six import string_types
from ansible.module_utils.six.moves import shlex_quote
from ansible.module_utils._text import to_text
from ansible.module_utils.common._collections_compat import MutableSequence
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.plugins.action import ActionBase
from ansible.plugins.loader import connection_loader
def _get_absolute_path(self, path):
    original_path = path
    if ':' in path or path.startswith('/'):
        return path
    if self._task._role is not None:
        path = self._loader.path_dwim_relative(self._task._role._role_path, 'files', path)
    else:
        path = self._loader.path_dwim_relative(self._loader.get_basedir(), 'files', path)
    if original_path and original_path[-1] == '/' and (path[-1] != '/'):
        path += '/'
    return path