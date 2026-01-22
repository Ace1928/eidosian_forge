from __future__ import (absolute_import, division, print_function)
import shlex
import ansible.module_utils.compat.typing as t
from ansible.module_utils.facts.utils import get_file_content
from ansible.module_utils.facts.collector import BaseFactCollector
def _parse_proc_cmdline_facts(self, data):
    cmdline_dict = {}
    try:
        for piece in shlex.split(data, posix=False):
            item = piece.split('=', 1)
            if len(item) == 1:
                cmdline_dict[item[0]] = True
            elif item[0] in cmdline_dict:
                if isinstance(cmdline_dict[item[0]], list):
                    cmdline_dict[item[0]].append(item[1])
                else:
                    new_list = [cmdline_dict[item[0]], item[1]]
                    cmdline_dict[item[0]] = new_list
            else:
                cmdline_dict[item[0]] = item[1]
    except ValueError:
        pass
    return cmdline_dict