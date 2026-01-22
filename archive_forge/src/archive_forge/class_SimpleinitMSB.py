from __future__ import absolute_import, division, print_function
import os
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.service import daemonize
class SimpleinitMSB(object):
    """
    Main simpleinit-msb service manipulation class
    """

    def __init__(self, module):
        self.module = module
        self.name = module.params['name']
        self.state = module.params['state']
        self.enable = module.params['enabled']
        self.changed = False
        self.running = None
        self.action = None
        self.telinit_cmd = None
        self.svc_change = False

    def execute_command(self, cmd, daemon=False):
        if not daemon:
            return self.module.run_command(cmd)
        else:
            return daemonize(self.module, cmd)

    def check_service_changed(self):
        if self.state and self.running is None:
            self.module.fail_json(msg='failed determining service state, possible typo of service name?')
        if not self.running and self.state in ['started', 'running', 'reloaded']:
            self.svc_change = True
        elif self.running and self.state in ['stopped', 'reloaded']:
            self.svc_change = True
        elif self.state == 'restarted':
            self.svc_change = True
        if self.module.check_mode and self.svc_change:
            self.module.exit_json(changed=True, msg='service state changed')

    def modify_service_state(self):
        if self.svc_change:
            if self.state in ['started', 'running']:
                self.action = 'start'
            elif not self.running and self.state == 'reloaded':
                self.action = 'start'
            elif self.state == 'stopped':
                self.action = 'stop'
            elif self.state == 'reloaded':
                self.action = 'reload'
            elif self.state == 'restarted':
                self.action = 'restart'
            if self.module.check_mode:
                self.module.exit_json(changed=True, msg='changing service state')
            return self.service_control()
        else:
            rc = 0
            err = ''
            out = ''
            return (rc, out, err)

    def get_service_tools(self):
        paths = ['/sbin', '/usr/sbin', '/bin', '/usr/bin']
        binaries = ['telinit']
        location = dict()
        for binary in binaries:
            location[binary] = self.module.get_bin_path(binary, opt_dirs=paths)
        if location.get('telinit', False) and os.path.exists('/etc/init.d/smgl_init'):
            self.telinit_cmd = location['telinit']
        if self.telinit_cmd is None:
            self.module.fail_json(msg='cannot find telinit script for simpleinit-msb, aborting...')

    def get_service_status(self):
        self.action = 'status'
        rc, status_stdout, status_stderr = self.service_control()
        if self.running is None and status_stdout.count('\n') <= 1:
            cleanout = status_stdout.lower().replace(self.name.lower(), '')
            if 'is not running' in cleanout:
                self.running = False
            elif 'is running' in cleanout:
                self.running = True
        return self.running

    def service_enable(self):
        if not self.enable ^ self.service_enabled():
            return
        action = 'boot' + ('enable' if self.enable else 'disable')
        rc, out, err = self.execute_command('%s %s %s' % (self.telinit_cmd, action, self.name))
        self.changed = True
        for line in err.splitlines():
            if self.enable and line.find('already enabled') != -1:
                self.changed = False
                break
            if not self.enable and line.find('already disabled') != -1:
                self.changed = False
                break
        if not self.changed:
            return
        return (rc, out, err)

    def service_enabled(self):
        self.service_exists()
        rc, out, err = self.execute_command('%s %sd' % (self.telinit_cmd, self.enable))
        service_enabled = False if self.enable else True
        rex = re.compile('^%s$' % self.name)
        for line in out.splitlines():
            if rex.match(line):
                service_enabled = True if self.enable else False
                break
        return service_enabled

    def service_exists(self):
        rc, out, err = self.execute_command('%s list' % self.telinit_cmd)
        service_exists = False
        rex = re.compile('^\\w+\\s+%s$' % self.name)
        for line in out.splitlines():
            if rex.match(line):
                service_exists = True
                break
        if not service_exists:
            self.module.fail_json(msg='telinit could not find the requested service: %s' % self.name)

    def service_control(self):
        self.service_exists()
        svc_cmd = '%s run %s' % (self.telinit_cmd, self.name)
        rc_state, stdout, stderr = self.execute_command('%s %s' % (svc_cmd, self.action), daemon=True)
        return (rc_state, stdout, stderr)