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
class SmartOSTimezone(Timezone):
    """This is a Timezone manipulation class for SmartOS instances.

    It uses the C(sm-set-timezone) utility to set the timezone, and
    inspects C(/etc/default/init) to determine the current timezone.

    NB: A zone needs to be rebooted in order for the change to be
    activated.
    """

    def __init__(self, module):
        super(SmartOSTimezone, self).__init__(module)
        self.settimezone = self.module.get_bin_path('sm-set-timezone', required=False)
        if not self.settimezone:
            module.fail_json(msg='sm-set-timezone not found. Make sure the smtools package is installed.')

    def get(self, key, phase):
        """Lookup the current timezone name in `/etc/default/init`. If anything else
        is requested, or if the TZ field is not set we fail.
        """
        if key == 'name':
            try:
                f = open('/etc/default/init', 'r')
                for line in f:
                    m = re.match('^TZ=(.*)$', line.strip())
                    if m:
                        return m.groups()[0]
            except Exception:
                self.module.fail_json(msg='Failed to read /etc/default/init')
        else:
            self.module.fail_json(msg='%s is not a supported option on target platform' % key)

    def set(self, key, value):
        """Set the requested timezone through sm-set-timezone, an invalid timezone name
        will be rejected and we have no further input validation to perform.
        """
        if key == 'name':
            cmd = 'sm-set-timezone %s' % value
            rc, stdout, stderr = self.module.run_command(cmd)
            if rc != 0:
                self.module.fail_json(msg=stderr)
            m = re.match('^\\* Changed (to)? timezone (to)? (%s).*' % value, stdout.splitlines()[1])
            if not (m and m.groups()[-1] == value):
                self.module.fail_json(msg='Failed to set timezone')
        else:
            self.module.fail_json(msg='%s is not a supported option on target platform' % key)