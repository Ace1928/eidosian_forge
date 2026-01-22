from __future__ import absolute_import, division, print_function
import os
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
class Sv(object):
    """
    Main class that handles daemontools, can be subclassed and overridden in case
    we want to use a 'derivative' like encore, s6, etc
    """

    def __init__(self, module):
        self.extra_paths = []
        self.report_vars = ['state', 'enabled', 'svc_full', 'src_full', 'pid', 'duration', 'full_state']
        self.module = module
        self.name = module.params['name']
        self.service_dir = module.params['service_dir']
        self.service_src = module.params['service_src']
        self.enabled = None
        self.full_state = None
        self.state = None
        self.pid = None
        self.duration = None
        self.svc_cmd = module.get_bin_path('sv', opt_dirs=self.extra_paths, required=True)
        self.svstat_cmd = module.get_bin_path('sv', opt_dirs=self.extra_paths)
        self.svc_full = '/'.join([self.service_dir, self.name])
        self.src_full = '/'.join([self.service_src, self.name])
        self.enabled = os.path.lexists(self.svc_full)
        if self.enabled:
            self.get_status()
        else:
            self.state = 'stopped'

    def enable(self):
        if os.path.exists(self.src_full):
            try:
                os.symlink(self.src_full, self.svc_full)
            except OSError as e:
                self.module.fail_json(path=self.src_full, msg='Error while linking: %s' % to_native(e))
        else:
            self.module.fail_json(msg='Could not find source for service to enable (%s).' % self.src_full)

    def disable(self):
        self.execute_command([self.svc_cmd, 'force-stop', self.src_full])
        try:
            os.unlink(self.svc_full)
        except OSError as e:
            self.module.fail_json(path=self.svc_full, msg='Error while unlinking: %s' % to_native(e))

    def get_status(self):
        rc, out, err = self.execute_command([self.svstat_cmd, 'status', self.svc_full])
        if err is not None and err:
            self.full_state = self.state = err
        else:
            self.full_state = out
            full_state_no_logger = self.full_state.split('; ')[0]
            m = re.search('\\(pid (\\d+)\\)', full_state_no_logger)
            if m:
                self.pid = m.group(1)
            m = re.search(' (\\d+)s', full_state_no_logger)
            if m:
                self.duration = m.group(1)
            if re.search('^run:', full_state_no_logger):
                self.state = 'started'
            elif re.search('^down:', full_state_no_logger):
                self.state = 'stopped'
            else:
                self.state = 'unknown'
                return

    def started(self):
        return self.start()

    def start(self):
        return self.execute_command([self.svc_cmd, 'start', self.svc_full])

    def stopped(self):
        return self.stop()

    def stop(self):
        return self.execute_command([self.svc_cmd, 'stop', self.svc_full])

    def once(self):
        return self.execute_command([self.svc_cmd, 'once', self.svc_full])

    def reloaded(self):
        return self.reload()

    def reload(self):
        return self.execute_command([self.svc_cmd, 'reload', self.svc_full])

    def restarted(self):
        return self.restart()

    def restart(self):
        return self.execute_command([self.svc_cmd, 'restart', self.svc_full])

    def killed(self):
        return self.kill()

    def kill(self):
        return self.execute_command([self.svc_cmd, 'force-stop', self.svc_full])

    def execute_command(self, cmd):
        try:
            rc, out, err = self.module.run_command(cmd)
        except Exception as e:
            self.module.fail_json(msg='failed to execute: %s' % to_native(e))
        return (rc, out, err)

    def report(self):
        self.get_status()
        states = {}
        for k in self.report_vars:
            states[k] = self.__dict__[k]
        return states