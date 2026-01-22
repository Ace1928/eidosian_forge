from __future__ import (absolute_import, division, print_function)
from subprocess import Popen, PIPE
from ansible.errors import AnsibleError
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.plugins.lookup import LookupBase
class LPass(object):

    def __init__(self, path='lpass'):
        self._cli_path = path

    @property
    def cli_path(self):
        return self._cli_path

    @property
    def logged_in(self):
        out, err = self._run(self._build_args('logout'), stdin='n\n', expected_rc=1)
        return err.startswith('Are you sure you would like to log out?')

    def _run(self, args, stdin=None, expected_rc=0):
        p = Popen([self.cli_path] + args, stdout=PIPE, stderr=PIPE, stdin=PIPE)
        out, err = p.communicate(to_bytes(stdin))
        rc = p.wait()
        if rc != expected_rc:
            raise LPassException(err)
        return (to_text(out, errors='surrogate_or_strict'), to_text(err, errors='surrogate_or_strict'))

    def _build_args(self, command, args=None):
        if args is None:
            args = []
        args = [command] + args
        args += ['--color=never']
        return args

    def get_field(self, key, field):
        if field in ['username', 'password', 'url', 'notes', 'id', 'name']:
            out, err = self._run(self._build_args('show', ['--{0}'.format(field), key]))
        else:
            out, err = self._run(self._build_args('show', ['--field={0}'.format(field), key]))
        return out.strip()