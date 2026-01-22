from __future__ import absolute_import, division, print_function
import time
import re
from collections import namedtuple
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import python_2_unicode_compatible
class Monit(object):

    def __init__(self, module, monit_bin_path, service_name, timeout):
        self.module = module
        self.monit_bin_path = monit_bin_path
        self.process_name = service_name
        self.timeout = timeout
        self._monit_version = None
        self._raw_version = None
        self._status_change_retry_count = 6

    def monit_version(self):
        if self._monit_version is None:
            self._raw_version, version = self._get_monit_version()
            self._monit_version = (version[0], version[1])
        return self._monit_version

    def _get_monit_version(self):
        rc, out, err = self.module.run_command([self.monit_bin_path, '-V'], check_rc=True)
        version_line = out.split('\n')[0]
        raw_version = re.search('([0-9]+\\.){1,2}([0-9]+)?', version_line).group()
        return (raw_version, tuple(map(int, raw_version.split('.'))))

    def exit_fail(self, msg, status=None, **kwargs):
        kwargs.update({'msg': msg, 'monit_version': self._raw_version, 'process_status': str(status) if status else None})
        self.module.fail_json(**kwargs)

    def exit_success(self, state):
        self.module.exit_json(changed=True, name=self.process_name, state=state)

    @property
    def command_args(self):
        return ['-B'] if self.monit_version() > (5, 18) else []

    def get_status(self, validate=False):
        """Return the status of the process in monit.

        :@param validate: Force monit to re-check the status of the process
        """
        monit_command = 'validate' if validate else 'status'
        check_rc = False if validate else True
        command = [self.monit_bin_path, monit_command] + self.command_args + [self.process_name]
        rc, out, err = self.module.run_command(command, check_rc=check_rc)
        return self._parse_status(out, err)

    def _parse_status(self, output, err):
        escaped_monit_services = '|'.join([re.escape(x) for x in MONIT_SERVICES])
        pattern = "(%s) '%s'" % (escaped_monit_services, re.escape(self.process_name))
        if not re.search(pattern, output, re.IGNORECASE):
            return Status.MISSING
        status_val = re.findall('^\\s*status\\s*([\\w\\- ]+)', output, re.MULTILINE)
        if not status_val:
            self.exit_fail('Unable to find process status', stdout=output, stderr=err)
        status_val = status_val[0].strip().upper()
        if ' | ' in status_val:
            status_val = status_val.split(' | ')[0]
        if ' - ' not in status_val:
            status_val = status_val.replace(' ', '_')
            return getattr(Status, status_val)
        else:
            status_val, substatus = status_val.split(' - ')
            action, state = substatus.split()
            if action in ['START', 'INITIALIZING', 'RESTART', 'MONITOR']:
                status = Status.OK
            else:
                status = Status.NOT_MONITORED
            if state == 'pending':
                status = status.pending()
            return status

    def is_process_present(self):
        command = [self.monit_bin_path, 'summary'] + self.command_args
        rc, out, err = self.module.run_command(command, check_rc=True)
        return bool(re.findall('\\b%s\\b' % self.process_name, out))

    def is_process_running(self):
        return self.get_status().is_ok

    def run_command(self, command):
        """Runs a monit command, and returns the new status."""
        return self.module.run_command([self.monit_bin_path, command, self.process_name], check_rc=True)

    def wait_for_status_change(self, current_status):
        running_status = self.get_status()
        if running_status.value != current_status.value or current_status.value == StatusValue.EXECUTION_FAILED:
            return running_status
        loop_count = 0
        while running_status.value == current_status.value:
            if loop_count >= self._status_change_retry_count:
                self.exit_fail('waited too long for monit to change state', running_status)
            loop_count += 1
            time.sleep(0.5)
            validate = loop_count % 2 == 0
            running_status = self.get_status(validate)
        return running_status

    def wait_for_monit_to_stop_pending(self, current_status=None):
        """Fails this run if there is no status or it's pending/initializing for timeout"""
        timeout_time = time.time() + self.timeout
        if not current_status:
            current_status = self.get_status()
        waiting_status = [StatusValue.MISSING, StatusValue.INITIALIZING, StatusValue.DOES_NOT_EXIST]
        while current_status.is_pending or current_status.value in waiting_status:
            if time.time() >= timeout_time:
                self.exit_fail('waited too long for "pending", or "initiating" status to go away', current_status)
            time.sleep(5)
            current_status = self.get_status(validate=True)
        return current_status

    def reload(self):
        rc, out, err = self.module.run_command([self.monit_bin_path, 'reload'])
        if rc != 0:
            self.exit_fail('monit reload failed', stdout=out, stderr=err)
        self.exit_success(state='reloaded')

    def present(self):
        self.run_command('reload')
        timeout_time = time.time() + self.timeout
        while not self.is_process_present():
            if time.time() >= timeout_time:
                self.exit_fail('waited too long for process to become "present"')
            time.sleep(5)
        self.exit_success(state='present')

    def change_state(self, state, expected_status, invert_expected=None):
        current_status = self.get_status()
        self.run_command(STATE_COMMAND_MAP[state])
        status = self.wait_for_status_change(current_status)
        status = self.wait_for_monit_to_stop_pending(status)
        status_match = status.value == expected_status.value
        if invert_expected:
            status_match = not status_match
        if status_match:
            self.exit_success(state=state)
        self.exit_fail('%s process not %s' % (self.process_name, state), status)

    def stop(self):
        self.change_state('stopped', Status.NOT_MONITORED)

    def unmonitor(self):
        self.change_state('unmonitored', Status.NOT_MONITORED)

    def restart(self):
        self.change_state('restarted', Status.OK)

    def start(self):
        self.change_state('started', Status.OK)

    def monitor(self):
        self.change_state('monitored', Status.NOT_MONITORED, invert_expected=True)