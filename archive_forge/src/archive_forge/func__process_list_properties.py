from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.module_helper import ModuleHelper
from ansible_collections.community.general.plugins.module_utils.xfconf import xfconf_runner
def _process_list_properties(self, rc, out, err):
    return out.splitlines()