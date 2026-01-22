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
def _startnewline(self, prompt=None, buf=None):
    buf = buf or []
    if not prompt and self.prompted:
        prompt = self.PROMPT
    if isinstance(buf, str):
        buf = list(buf)
    if self.chan:
        self.buf = buf
        if prompt:
            self.chan.send('\n\r' + prompt + ''.join(buf))
            self.curpos = len(prompt) + len(buf)
            self.prompted = True
        else:
            self.chan.send('\n\r' + ''.join(buf))
            self.curpos = len(buf)
            self.prompted = False