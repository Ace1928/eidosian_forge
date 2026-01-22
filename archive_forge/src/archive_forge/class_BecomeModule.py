from __future__ import (absolute_import, division, print_function)
from ansible.plugins.become import BecomeBase
class BecomeModule(BecomeBase):
    name = 'runas'

    def build_become_command(self, cmd, shell):
        return cmd