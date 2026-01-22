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
def _patch_update_module(self, module, task_vars):
    """Update a module instance, replacing it's AnsibleModule
        with one that doesn't load params

        :param module: An loaded module
        :type module: A module file that was loaded
        :param task_vars: The vars provided to the task
        :type task_vars: dict
        """
    import copy
    from ansible.module_utils.basic import AnsibleModule as _AnsibleModule

    class PatchedAnsibleModule(_AnsibleModule):

        def _load_params(self):
            pass
    self._update_module_args(self._task.action, self._task.args, task_vars)
    PatchedAnsibleModule.params = copy.deepcopy(self._task.args)
    module.AnsibleModule = PatchedAnsibleModule