from __future__ import absolute_import, division, print_function
import re
from ansible_collections.community.general.plugins.module_utils.cmd_runner import CmdRunner, cmd_runner_fmt as fmt
from ansible_collections.community.general.plugins.module_utils.module_helper import ModuleHelper, ModuleHelperException
class UnsupportedLocale(ModuleHelperException):
    pass