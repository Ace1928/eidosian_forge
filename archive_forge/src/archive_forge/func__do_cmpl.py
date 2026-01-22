from copy import copy
import logging
import os.path
import sys
import paramiko
from os_ken import version
from os_ken.lib import hub
from os_ken.services.protocols.bgp.base import Activity
from os_ken.services.protocols.bgp.operator.command import Command
from os_ken.services.protocols.bgp.operator.command import CommandsResponse
from os_ken.services.protocols.bgp.operator.command import STATUS_OK
from os_ken.services.protocols.bgp.operator.commands.root import RootCmd
from os_ken.services.protocols.bgp.operator.internal_api import InternalApi
def _do_cmpl(self, buf, is_exec=False):
    cmpleter = self.root
    is_spaced = buf[-1] == ' ' if len(buf) > 0 else False
    cmds = [tkn.strip() for tkn in ''.join(buf).split()]
    ret = []
    for i, cmd in enumerate(cmds):
        subcmds = cmpleter.subcommands
        matches = [x for x in subcmds.keys() if x.startswith(cmd)]
        if len(matches) == 1:
            cmpled_cmd = matches[0]
            cmpleter = subcmds[cmpled_cmd](self.api)
            if is_exec:
                ret.append(cmpled_cmd)
                continue
            if i + 1 == len(cmds):
                if is_spaced:
                    result, cmd = cmpleter('?')
                    result = result.value.replace('\n', '\n\r').rstrip()
                    self.prompted = False
                    buf = copy(buf)
                    self._startnewline(buf=result)
                    self.prompted = True
                    self._startnewline(buf=buf)
                else:
                    self.buf = buf[:-1 * len(cmd)] + list(cmpled_cmd + ' ')
                    self.curpos += len(cmpled_cmd) - len(cmd) + 1
                    self._refreshline()
        else:
            self.prompted = False
            buf = copy(self.buf)
            if len(matches) == 0:
                if cmpleter.param_help_msg:
                    self.prompted = True
                    ret.append(cmd)
                    continue
                else:
                    self._startnewline(buf='Error: Not implemented')
            elif i + 1 < len(cmds):
                self._startnewline(buf='Error: Ambiguous command')
            else:
                self._startnewline(buf=', '.join(matches))
            ret = []
            self.prompted = True
            if not is_exec:
                self._startnewline(buf=buf)
            break
    return ret