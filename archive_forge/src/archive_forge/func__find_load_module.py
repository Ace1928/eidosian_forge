from __future__ import absolute_import, division, print_function
import os
import re
import time
from ansible.errors import AnsibleError
from ansible.module_utils._text import to_text
from ansible.module_utils.six import PY3
from ansible.module_utils.six.moves.urllib.parse import urlsplit
from ansible.plugins.action.normal import ActionModule as _ActionModule
from ansible.utils.display import Display
from ansible.utils.hashing import checksum, checksum_s
def _find_load_module(self):
    """Use the task action to find a module
        and import it.

        :return filename: The module's filename
        :rtype filename: str
        :return module: The loaded module file
        :rtype module: module
        """
    import importlib
    mloadr = self._shared_loader_obj.module_loader
    try:
        context = mloadr.find_plugin_with_context(self._task.action, collection_list=self._task.collections)
        filename = context.plugin_resolved_path
        module = importlib.import_module(context.plugin_resolved_name)
    except AttributeError:
        fullname, filename = mloadr.find_plugin_with_name(self._task.action, collection_list=self._task.collections)
        module = importlib.import_module(fullname)
    return (filename, module)