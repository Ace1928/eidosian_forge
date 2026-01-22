from __future__ import absolute_import, division, print_function
import re
from ansible_collections.community.general.plugins.module_utils.cmd_runner import CmdRunner, cmd_runner_fmt as fmt
from ansible_collections.community.general.plugins.module_utils.module_helper import ModuleHelper, ModuleHelperException
def _list_collections(self):
    return self._list_element('collection', self._RE_LIST_PATH, self._RE_LIST_COLL)