from __future__ import (absolute_import, division, print_function)
import shlex
import ansible.module_utils.compat.typing as t
from ansible.module_utils.facts.utils import get_file_content
from ansible.module_utils.facts.collector import BaseFactCollector
class CmdLineFactCollector(BaseFactCollector):
    name = 'cmdline'
    _fact_ids = set()

    def _get_proc_cmdline(self):
        return get_file_content('/proc/cmdline')

    def _parse_proc_cmdline(self, data):
        cmdline_dict = {}
        try:
            for piece in shlex.split(data, posix=False):
                item = piece.split('=', 1)
                if len(item) == 1:
                    cmdline_dict[item[0]] = True
                else:
                    cmdline_dict[item[0]] = item[1]
        except ValueError:
            pass
        return cmdline_dict

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

    def collect(self, module=None, collected_facts=None):
        cmdline_facts = {}
        data = self._get_proc_cmdline()
        if not data:
            return cmdline_facts
        cmdline_facts['cmdline'] = self._parse_proc_cmdline(data)
        cmdline_facts['proc_cmdline'] = self._parse_proc_cmdline_facts(data)
        return cmdline_facts