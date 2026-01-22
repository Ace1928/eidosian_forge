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
class LinuxService(Service):
    """
    This is the Linux Service manipulation class - it is currently supporting
    a mixture of binaries and init scripts for controlling services started at
    boot, as well as for controlling the current state.
    """
    platform = 'Linux'
    distribution = None

    def get_service_tools(self):
        paths = ['/sbin', '/usr/sbin', '/bin', '/usr/bin']
        binaries = ['service', 'chkconfig', 'update-rc.d', 'rc-service', 'rc-update', 'initctl', 'systemctl', 'start', 'stop', 'restart', 'insserv']
        initpaths = ['/etc/init.d']
        location = dict()
        for binary in binaries:
            location[binary] = self.module.get_bin_path(binary, opt_dirs=paths)
        for initdir in initpaths:
            initscript = '%s/%s' % (initdir, self.name)
            if os.path.isfile(initscript):
                self.svc_initscript = initscript

        def check_systemd():
            if location.get('systemctl', False):
                for canary in ['/run/systemd/system/', '/dev/.run/systemd/', '/dev/.systemd/']:
                    if os.path.exists(canary):
                        return True
                try:
                    f = open('/proc/1/comm', 'r')
                except IOError:
                    return False
                for line in f:
                    if 'systemd' in line:
                        return True
            return False
        if check_systemd():
            self.__systemd_unit = self.name
            self.svc_cmd = location['systemctl']
            self.enable_cmd = location['systemctl']
        elif location.get('initctl', False) and os.path.exists('/etc/init/%s.conf' % self.name):
            self.enable_cmd = location['initctl']
            self.upstart_version = LooseVersion('0.0.0')
            try:
                version_re = re.compile('\\(upstart (.*)\\)')
                rc, stdout, stderr = self.module.run_command('%s version' % location['initctl'])
                if rc == 0:
                    res = version_re.search(stdout)
                    if res:
                        self.upstart_version = LooseVersion(res.groups()[0])
            except Exception:
                pass
            self.svc_cmd = location['initctl']
        elif location.get('rc-service', False):
            self.svc_cmd = location['rc-service']
            self.enable_cmd = location['rc-update']
            return
        elif self.svc_initscript:
            if location.get('update-rc.d', False):
                self.enable_cmd = location['update-rc.d']
            elif location.get('insserv', None):
                self.enable_cmd = location['insserv']
            elif location.get('chkconfig', False):
                self.enable_cmd = location['chkconfig']
        if self.enable_cmd is None:
            fail_if_missing(self.module, False, self.name, msg='host')
        if self.svc_cmd is None and location.get('service', False):
            self.svc_cmd = location['service']
        if self.svc_cmd is None and (not self.svc_initscript):
            self.module.fail_json(msg="cannot find 'service' binary or init script for service,  possible typo in service name?, aborting")
        if location.get('initctl', False):
            self.svc_initctl = location['initctl']

    def get_systemd_service_enabled(self):

        def sysv_exists(name):
            script = '/etc/init.d/' + name
            return os.access(script, os.X_OK)

        def sysv_is_enabled(name):
            return bool(glob.glob('/etc/rc?.d/S??' + name))
        service_name = self.__systemd_unit
        rc, out, err = self.execute_command('%s is-enabled %s' % (self.enable_cmd, service_name))
        if rc == 0:
            return True
        elif out.startswith('disabled'):
            return False
        elif sysv_exists(service_name):
            return sysv_is_enabled(service_name)
        else:
            return False

    def get_systemd_status_dict(self):
        rc, out, err = self.execute_command("%s show '%s'" % (self.enable_cmd, self.__systemd_unit))
        if rc != 0:
            self.module.fail_json(msg='failure %d running systemctl show for %r: %s' % (rc, self.__systemd_unit, err))
        elif 'LoadState=not-found' in out:
            self.module.fail_json(msg='systemd could not find the requested service "%r": %s' % (self.__systemd_unit, err))
        key = None
        value_buffer = []
        status_dict = {}
        for line in out.splitlines():
            if '=' in line:
                if not key:
                    key, value = line.split('=', 1)
                    if value.lstrip().startswith('{'):
                        if value.rstrip().endswith('}'):
                            status_dict[key] = value
                            key = None
                        else:
                            value_buffer.append(value)
                    else:
                        status_dict[key] = value
                        key = None
                elif line.rstrip().endswith('}'):
                    status_dict[key] = '\n'.join(value_buffer)
                    key = None
                else:
                    value_buffer.append(value)
            else:
                value_buffer.append(value)
        return status_dict

    def get_systemd_service_status(self):
        d = self.get_systemd_status_dict()
        if d.get('ActiveState') == 'active':
            self.running = True
            self.crashed = False
        elif d.get('ActiveState') == 'failed':
            self.running = False
            self.crashed = True
        elif d.get('ActiveState') is None:
            self.module.fail_json(msg='No ActiveState value in systemctl show output for %r' % (self.__systemd_unit,))
        else:
            self.running = False
            self.crashed = False
        return self.running

    def get_service_status(self):
        if self.svc_cmd and self.svc_cmd.endswith('systemctl'):
            return self.get_systemd_service_status()
        self.action = 'status'
        rc, status_stdout, status_stderr = self.service_control()
        if self.svc_initctl and self.running is None:
            initctl_rc, initctl_status_stdout, initctl_status_stderr = self.execute_command('%s status %s %s' % (self.svc_initctl, self.name, self.arguments))
            if 'stop/waiting' in initctl_status_stdout:
                self.running = False
            elif 'start/running' in initctl_status_stdout:
                self.running = True
        if self.svc_cmd and self.svc_cmd.endswith('rc-service') and (self.running is None):
            openrc_rc, openrc_status_stdout, openrc_status_stderr = self.execute_command('%s %s status' % (self.svc_cmd, self.name))
            self.running = 'started' in openrc_status_stdout
            self.crashed = 'crashed' in openrc_status_stderr
        if self.running is None and rc in [1, 2, 3, 4, 69]:
            self.running = False
        if self.running is None and status_stdout.count('\n') <= 1:
            cleanout = status_stdout.lower().replace(self.name.lower(), '')
            if 'stop' in cleanout:
                self.running = False
            elif 'run' in cleanout:
                self.running = not 'not ' in cleanout
            elif 'start' in cleanout and 'not ' not in cleanout:
                self.running = True
            elif 'could not access pid file' in cleanout:
                self.running = False
            elif 'is dead and pid file exists' in cleanout:
                self.running = False
            elif 'dead but subsys locked' in cleanout:
                self.running = False
            elif 'dead but pid file exists' in cleanout:
                self.running = False
        if self.running is None and rc == 0:
            self.running = True
        if self.running is None:
            if self.name == 'iptables' and 'ACCEPT' in status_stdout:
                self.running = True
        return self.running

    def service_enable(self):
        if self.enable_cmd is None:
            self.module.fail_json(msg='cannot detect command to enable service %s, typo or init system potentially unknown' % self.name)
        self.changed = True
        action = None
        if self.enable_cmd.endswith('initctl'):

            def write_to_override_file(file_name, file_contents):
                override_file = open(file_name, 'w')
                override_file.write(file_contents)
                override_file.close()
            initpath = '/etc/init'
            if self.upstart_version >= LooseVersion('0.6.7'):
                manreg = re.compile('^manual\\s*$', re.M | re.I)
                config_line = 'manual\n'
            else:
                manreg = re.compile('^start on manual\\s*$', re.M | re.I)
                config_line = 'start on manual\n'
            conf_file_name = '%s/%s.conf' % (initpath, self.name)
            override_file_name = '%s/%s.override' % (initpath, self.name)
            with open(conf_file_name) as conf_file_fh:
                conf_file_content = conf_file_fh.read()
            if manreg.search(conf_file_content):
                self.module.fail_json(msg='manual stanza not supported in a .conf file')
            self.changed = False
            if os.path.exists(override_file_name):
                with open(override_file_name) as override_fh:
                    override_file_contents = override_fh.read()
                if self.enable and manreg.search(override_file_contents):
                    self.changed = True
                    override_state = manreg.sub('', override_file_contents)
                elif not self.enable and (not manreg.search(override_file_contents)):
                    self.changed = True
                    override_state = '\n'.join((override_file_contents, config_line))
                else:
                    pass
            elif not self.enable:
                self.changed = True
                override_state = config_line
            else:
                pass
            if self.module.check_mode:
                self.module.exit_json(changed=self.changed)
            if self.changed:
                try:
                    write_to_override_file(override_file_name, override_state)
                except Exception:
                    self.module.fail_json(msg='Could not modify override file')
            return
        if self.enable_cmd.endswith('chkconfig'):
            if self.enable:
                action = 'on'
            else:
                action = 'off'
            rc, out, err = self.execute_command('%s --list %s' % (self.enable_cmd, self.name))
            if 'chkconfig --add %s' % self.name in err:
                self.execute_command('%s --add %s' % (self.enable_cmd, self.name))
                rc, out, err = self.execute_command('%s --list %s' % (self.enable_cmd, self.name))
            if self.name not in out:
                self.module.fail_json(msg='service %s does not support chkconfig' % self.name)
            if '3:%s' % action in out and '5:%s' % action in out:
                self.changed = False
                return
        if self.enable_cmd.endswith('systemctl'):
            if self.enable:
                action = 'enable'
            else:
                action = 'disable'
            service_enabled = self.get_systemd_service_enabled()
            if self.enable == service_enabled:
                self.changed = False
                return
        if self.enable_cmd.endswith('rc-update'):
            if self.enable:
                action = 'add'
            else:
                action = 'delete'
            rc, out, err = self.execute_command('%s show' % self.enable_cmd)
            for line in out.splitlines():
                service_name, runlevels = line.split('|')
                service_name = service_name.strip()
                if service_name != self.name:
                    continue
                runlevels = re.split('\\s+', runlevels)
                if self.enable and self.runlevel in runlevels:
                    self.changed = False
                elif not self.enable and self.runlevel not in runlevels:
                    self.changed = False
                break
            else:
                if not self.enable:
                    self.changed = False
            if not self.changed:
                return
        if self.enable_cmd.endswith('update-rc.d'):
            enabled = False
            slinks = glob.glob('/etc/rc?.d/S??' + self.name)
            if slinks:
                enabled = True
            if self.enable != enabled:
                self.changed = True
                if self.enable:
                    action = 'enable'
                    klinks = glob.glob('/etc/rc?.d/K??' + self.name)
                    if not klinks:
                        if not self.module.check_mode:
                            rc, out, err = self.execute_command('%s %s defaults' % (self.enable_cmd, self.name))
                            if rc != 0:
                                if err:
                                    self.module.fail_json(msg=err)
                                else:
                                    self.module.fail_json(msg=out) % (self.enable_cmd, self.name, action)
                else:
                    action = 'disable'
                if not self.module.check_mode:
                    rc, out, err = self.execute_command('%s %s %s' % (self.enable_cmd, self.name, action))
                    if rc != 0:
                        if err:
                            self.module.fail_json(msg=err)
                        else:
                            self.module.fail_json(msg=out) % (self.enable_cmd, self.name, action)
            else:
                self.changed = False
            return
        if self.enable_cmd.endswith('insserv'):
            if self.enable:
                rc, out, err = self.execute_command('%s -n -v %s' % (self.enable_cmd, self.name))
            else:
                rc, out, err = self.execute_command('%s -n -r -v %s' % (self.enable_cmd, self.name))
            self.changed = False
            for line in err.splitlines():
                if self.enable and line.find('enable service') != -1:
                    self.changed = True
                    break
                if not self.enable and line.find('remove service') != -1:
                    self.changed = True
                    break
            if self.module.check_mode:
                self.module.exit_json(changed=self.changed)
            if not self.changed:
                return
            if self.enable:
                rc, out, err = self.execute_command('%s %s' % (self.enable_cmd, self.name))
                if rc != 0 or err != '':
                    self.module.fail_json(msg='Failed to install service. rc: %s, out: %s, err: %s' % (rc, out, err))
                return (rc, out, err)
            else:
                rc, out, err = self.execute_command('%s -r %s' % (self.enable_cmd, self.name))
                if rc != 0 or err != '':
                    self.module.fail_json(msg='Failed to remove service. rc: %s, out: %s, err: %s' % (rc, out, err))
                return (rc, out, err)
        self.changed = True
        if self.enable_cmd.endswith('rc-update'):
            args = (self.enable_cmd, action, self.name + ' ' + self.runlevel)
        elif self.enable_cmd.endswith('systemctl'):
            args = (self.enable_cmd, action, self.__systemd_unit)
        else:
            args = (self.enable_cmd, self.name, action)
        if self.module.check_mode:
            self.module.exit_json(changed=self.changed)
        rc, out, err = self.execute_command('%s %s %s' % args)
        if rc != 0:
            if err:
                self.module.fail_json(msg='Error when trying to %s %s: rc=%s %s' % (action, self.name, rc, err))
            else:
                self.module.fail_json(msg='Failure for %s %s: rc=%s %s' % (action, self.name, rc, out))
        return (rc, out, err)

    def service_control(self):
        svc_cmd = ''
        arguments = self.arguments
        if self.svc_cmd:
            if not self.svc_cmd.endswith('systemctl'):
                if self.svc_cmd.endswith('initctl'):
                    svc_cmd = self.svc_cmd
                    arguments = '%s %s' % (self.name, arguments)
                else:
                    svc_cmd = '%s %s' % (self.svc_cmd, self.name)
            else:
                svc_cmd = self.svc_cmd
                arguments = '%s %s' % (self.__systemd_unit, arguments)
        elif self.svc_cmd is None and self.svc_initscript:
            svc_cmd = '%s' % self.svc_initscript
        if self.svc_cmd and self.svc_cmd.endswith('rc-service') and (self.action == 'start') and self.crashed:
            self.execute_command('%s zap' % svc_cmd, daemonize=True)
        if self.action != 'restart':
            if svc_cmd != '':
                rc_state, stdout, stderr = self.execute_command('%s %s %s' % (svc_cmd, self.action, arguments), daemonize=True)
            else:
                rc_state, stdout, stderr = self.execute_command('%s %s %s' % (self.action, self.name, arguments), daemonize=True)
        elif self.svc_cmd and self.svc_cmd.endswith('rc-service'):
            rc_state, stdout, stderr = self.execute_command('%s %s %s' % (svc_cmd, self.action, arguments), daemonize=True)
        else:
            if svc_cmd != '':
                rc1, stdout1, stderr1 = self.execute_command('%s %s %s' % (svc_cmd, 'stop', arguments), daemonize=True)
            else:
                rc1, stdout1, stderr1 = self.execute_command('%s %s %s' % ('stop', self.name, arguments), daemonize=True)
            if self.sleep:
                time.sleep(self.sleep)
            if svc_cmd != '':
                rc2, stdout2, stderr2 = self.execute_command('%s %s %s' % (svc_cmd, 'start', arguments), daemonize=True)
            else:
                rc2, stdout2, stderr2 = self.execute_command('%s %s %s' % ('start', self.name, arguments), daemonize=True)
            if rc1 != 0 and rc2 == 0:
                rc_state = rc2
                stdout = stdout2
                stderr = stderr2
            else:
                rc_state = rc1 + rc2
                stdout = stdout1 + stdout2
                stderr = stderr1 + stderr2
        return (rc_state, stdout, stderr)