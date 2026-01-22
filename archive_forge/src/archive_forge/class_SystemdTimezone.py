from __future__ import absolute_import, division, print_function
import errno
import os
import platform
import random
import re
import string
import filecmp
from ansible.module_utils.basic import AnsibleModule, get_distribution
from ansible.module_utils.six import iteritems
class SystemdTimezone(Timezone):
    """This is a Timezone manipulation class for systemd-powered Linux.

    It uses the `timedatectl` command to check/set all arguments.
    """
    regexps = dict(hwclock=re.compile('^\\s*RTC in local TZ\\s*:\\s*([^\\s]+)', re.MULTILINE), name=re.compile('^\\s*Time ?zone\\s*:\\s*([^\\s]+)', re.MULTILINE))
    subcmds = dict(hwclock='set-local-rtc', name='set-timezone')

    def __init__(self, module):
        super(SystemdTimezone, self).__init__(module)
        self.timedatectl = module.get_bin_path('timedatectl', required=True)
        self.status = dict()
        if 'name' in self.value:
            self._verify_timezone()

    def _get_status(self, phase):
        if phase not in self.status:
            self.status[phase] = self.execute(self.timedatectl, 'status')
        return self.status[phase]

    def get(self, key, phase):
        status = self._get_status(phase)
        value = self.regexps[key].search(status).group(1)
        if key == 'hwclock':
            if self.module.boolean(value):
                value = 'local'
            else:
                value = 'UTC'
        return value

    def set(self, key, value):
        if key == 'hwclock':
            if value == 'local':
                value = 'yes'
            else:
                value = 'no'
        self.execute(self.timedatectl, self.subcmds[key], value, log=True)