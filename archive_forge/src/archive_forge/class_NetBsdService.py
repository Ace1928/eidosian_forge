from __future__ import absolute_import, division, print_function
import glob
import json
import os
import platform
import re
import select
import shlex
import subprocess
import tempfile
import time
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.common.sys_info import get_platform_subclass
from ansible.module_utils.service import fail_if_missing
from ansible.module_utils.six import PY2, b
class NetBsdService(Service):
    """
    This is the NetBSD Service manipulation class - it uses the /etc/rc.conf
    file for controlling services started at boot, check status and perform
    direct service manipulation. Init scripts in /etc/rc.d are used for
    controlling services (start/stop) as well as for controlling the current
    state.
    """
    platform = 'NetBSD'
    distribution = None

    def get_service_tools(self):
        initpaths = ['/etc/rc.d']
        for initdir in initpaths:
            initscript = '%s/%s' % (initdir, self.name)
            if os.path.isfile(initscript):
                self.svc_initscript = initscript
        if not self.svc_initscript:
            self.module.fail_json(msg='unable to find rc.d script')

    def service_enable(self):
        if self.enable:
            self.rcconf_value = 'YES'
        else:
            self.rcconf_value = 'NO'
        rcfiles = ['/etc/rc.conf']
        for rcfile in rcfiles:
            if os.path.isfile(rcfile):
                self.rcconf_file = rcfile
        self.rcconf_key = '%s' % self.name.replace('-', '_')
        return self.service_enable_rcconf()

    def get_service_status(self):
        self.svc_cmd = '%s' % self.svc_initscript
        rc, stdout, stderr = self.execute_command('%s %s' % (self.svc_cmd, 'onestatus'))
        if rc == 1:
            self.running = False
        elif rc == 0:
            self.running = True

    def service_control(self):
        if self.action == 'start':
            self.action = 'onestart'
        if self.action == 'stop':
            self.action = 'onestop'
        self.svc_cmd = '%s' % self.svc_initscript
        return self.execute_command('%s %s' % (self.svc_cmd, self.action), daemonize=True)