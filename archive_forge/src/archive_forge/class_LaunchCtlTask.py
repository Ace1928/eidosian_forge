from __future__ import absolute_import, division, print_function
import os
import plistlib
from abc import ABCMeta, abstractmethod
from time import sleep
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
class LaunchCtlTask(object):
    __metaclass__ = ABCMeta
    WAITING_TIME = 5

    def __init__(self, module, service, plist):
        self._module = module
        self._service = service
        self._plist = plist
        self._launch = self._module.get_bin_path('launchctl', True)

    def run(self):
        """Runs a launchd command like 'load', 'unload', 'start', 'stop', etc.
        and returns the new state and pid.
        """
        self.runCommand()
        return self.get_state()

    @abstractmethod
    def runCommand(self):
        pass

    def get_state(self):
        rc, out, err = self._launchctl('list')
        if rc != 0:
            self._module.fail_json(msg='Failed to get status of %s' % self._launch)
        state = ServiceState.UNLOADED
        service_pid = '-'
        status_code = None
        for line in out.splitlines():
            if line.strip():
                pid, last_exit_code, label = line.split('\t')
                if label.strip() == self._service:
                    service_pid = pid
                    status_code = last_exit_code
                    if last_exit_code not in ['0', '-2', '-3', '-9', '-15']:
                        state = ServiceState.UNKNOWN
                    elif pid != '-':
                        state = ServiceState.STARTED
                    else:
                        state = ServiceState.STOPPED
                    break
        return (state, service_pid, status_code, err)

    def start(self):
        rc, out, err = self._launchctl('start')
        sleep(self.WAITING_TIME)
        return (rc, out, err)

    def stop(self):
        rc, out, err = self._launchctl('stop')
        sleep(self.WAITING_TIME)
        return (rc, out, err)

    def restart(self):
        self.stop()
        return self.start()

    def reload(self):
        self.unload()
        return self.load()

    def load(self):
        return self._launchctl('load')

    def unload(self):
        return self._launchctl('unload')

    def _launchctl(self, command):
        service_or_plist = self._plist.get_file() if command in ['load', 'unload'] else self._service if command in ['start', 'stop'] else ''
        rc, out, err = self._module.run_command('%s %s %s' % (self._launch, command, service_or_plist))
        if rc != 0:
            msg = "Unable to %s '%s' (%s): '%s'" % (command, self._service, self._plist.get_file(), err)
            self._module.fail_json(msg=msg)
        return (rc, out, err)