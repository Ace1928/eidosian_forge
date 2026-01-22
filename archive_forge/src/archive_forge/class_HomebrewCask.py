from __future__ import absolute_import, division, print_function
import os
import re
import tempfile
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems, string_types
class HomebrewCask(object):
    """A class to manage Homebrew casks."""
    VALID_PATH_CHARS = '\n        \\w                  # alphanumeric characters (i.e., [a-zA-Z0-9_])\n        \\s                  # spaces\n        :                   # colons\n        {sep}               # the OS-specific path separator\n        .                   # dots\n        \\-                  # dashes\n    '.format(sep=os.path.sep)
    VALID_BREW_PATH_CHARS = '\n        \\w                  # alphanumeric characters (i.e., [a-zA-Z0-9_])\n        \\s                  # spaces\n        {sep}               # the OS-specific path separator\n        .                   # dots\n        \\-                  # dashes\n    '.format(sep=os.path.sep)
    VALID_CASK_CHARS = '\n        \\w                  # alphanumeric characters (i.e., [a-zA-Z0-9_])\n        .                   # dots\n        /                   # slash (for taps)\n        \\-                  # dashes\n        @                   # at symbol\n    '
    INVALID_PATH_REGEX = _create_regex_group_complement(VALID_PATH_CHARS)
    INVALID_BREW_PATH_REGEX = _create_regex_group_complement(VALID_BREW_PATH_CHARS)
    INVALID_CASK_REGEX = _create_regex_group_complement(VALID_CASK_CHARS)

    @classmethod
    def valid_path(cls, path):
        """
        `path` must be one of:
         - list of paths
         - a string containing only:
             - alphanumeric characters
             - dashes
             - dots
             - spaces
             - colons
             - os.path.sep
        """
        if isinstance(path, string_types):
            return not cls.INVALID_PATH_REGEX.search(path)
        try:
            iter(path)
        except TypeError:
            return False
        else:
            paths = path
            return all((cls.valid_brew_path(path_) for path_ in paths))

    @classmethod
    def valid_brew_path(cls, brew_path):
        """
        `brew_path` must be one of:
         - None
         - a string containing only:
             - alphanumeric characters
             - dashes
             - dots
             - spaces
             - os.path.sep
        """
        if brew_path is None:
            return True
        return isinstance(brew_path, string_types) and (not cls.INVALID_BREW_PATH_REGEX.search(brew_path))

    @classmethod
    def valid_cask(cls, cask):
        """A valid cask is either None or alphanumeric + backslashes."""
        if cask is None:
            return True
        return isinstance(cask, string_types) and (not cls.INVALID_CASK_REGEX.search(cask))

    @classmethod
    def valid_state(cls, state):
        """
        A valid state is one of:
            - installed
            - absent
        """
        if state is None:
            return True
        else:
            return isinstance(state, string_types) and state.lower() in ('installed', 'absent')

    @classmethod
    def valid_module(cls, module):
        """A valid module is an instance of AnsibleModule."""
        return isinstance(module, AnsibleModule)

    @property
    def module(self):
        return self._module

    @module.setter
    def module(self, module):
        if not self.valid_module(module):
            self._module = None
            self.failed = True
            self.message = 'Invalid module: {0}.'.format(module)
            raise HomebrewCaskException(self.message)
        else:
            self._module = module
            return module

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, path):
        if not self.valid_path(path):
            self._path = []
            self.failed = True
            self.message = 'Invalid path: {0}.'.format(path)
            raise HomebrewCaskException(self.message)
        else:
            if isinstance(path, string_types):
                self._path = path.split(':')
            else:
                self._path = path
            return path

    @property
    def brew_path(self):
        return self._brew_path

    @brew_path.setter
    def brew_path(self, brew_path):
        if not self.valid_brew_path(brew_path):
            self._brew_path = None
            self.failed = True
            self.message = 'Invalid brew_path: {0}.'.format(brew_path)
            raise HomebrewCaskException(self.message)
        else:
            self._brew_path = brew_path
            return brew_path

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, params):
        self._params = self.module.params
        return self._params

    @property
    def current_cask(self):
        return self._current_cask

    @current_cask.setter
    def current_cask(self, cask):
        if not self.valid_cask(cask):
            self._current_cask = None
            self.failed = True
            self.message = 'Invalid cask: {0}.'.format(cask)
            raise HomebrewCaskException(self.message)
        else:
            self._current_cask = cask
            return cask

    @property
    def brew_version(self):
        try:
            return self._brew_version
        except AttributeError:
            return None

    @brew_version.setter
    def brew_version(self, brew_version):
        self._brew_version = brew_version

    def __init__(self, module, path=path, casks=None, state=None, sudo_password=None, update_homebrew=False, install_options=None, accept_external_apps=False, upgrade_all=False, greedy=False):
        if not install_options:
            install_options = list()
        self._setup_status_vars()
        self._setup_instance_vars(module=module, path=path, casks=casks, state=state, sudo_password=sudo_password, update_homebrew=update_homebrew, install_options=install_options, accept_external_apps=accept_external_apps, upgrade_all=upgrade_all, greedy=greedy)
        self._prep()

    def _setup_status_vars(self):
        self.failed = False
        self.changed = False
        self.changed_count = 0
        self.unchanged_count = 0
        self.message = ''

    def _setup_instance_vars(self, **kwargs):
        for key, val in iteritems(kwargs):
            setattr(self, key, val)

    def _prep(self):
        self._prep_brew_path()

    def _prep_brew_path(self):
        if not self.module:
            self.brew_path = None
            self.failed = True
            self.message = 'AnsibleModule not set.'
            raise HomebrewCaskException(self.message)
        self.brew_path = self.module.get_bin_path('brew', required=True, opt_dirs=self.path)
        if not self.brew_path:
            self.brew_path = None
            self.failed = True
            self.message = 'Unable to locate homebrew executable.'
            raise HomebrewCaskException('Unable to locate homebrew executable.')
        return self.brew_path

    def _status(self):
        return (self.failed, self.changed, self.message)

    def run(self):
        try:
            self._run()
        except HomebrewCaskException:
            pass
        if not self.failed and self.changed_count + self.unchanged_count > 1:
            self.message = 'Changed: %d, Unchanged: %d' % (self.changed_count, self.unchanged_count)
        failed, changed, message = self._status()
        return (failed, changed, message)

    def _current_cask_is_outdated(self):
        if not self.valid_cask(self.current_cask):
            return False
        if self._brew_cask_command_is_deprecated():
            base_opts = [self.brew_path, 'outdated', '--cask']
        else:
            base_opts = [self.brew_path, 'cask', 'outdated']
        cask_is_outdated_command = base_opts + (['--greedy'] if self.greedy else []) + [self.current_cask]
        rc, out, err = self.module.run_command(cask_is_outdated_command)
        return out != ''

    def _current_cask_is_installed(self):
        if not self.valid_cask(self.current_cask):
            self.failed = True
            self.message = 'Invalid cask: {0}.'.format(self.current_cask)
            raise HomebrewCaskException(self.message)
        if self._brew_cask_command_is_deprecated():
            base_opts = [self.brew_path, 'list', '--cask']
        else:
            base_opts = [self.brew_path, 'cask', 'list']
        cmd = base_opts + [self.current_cask]
        rc, out, err = self.module.run_command(cmd)
        if rc == 0:
            return True
        else:
            return False

    def _get_brew_version(self):
        if self.brew_version:
            return self.brew_version
        cmd = [self.brew_path, '--version']
        rc, out, err = self.module.run_command(cmd, check_rc=True)
        version = out.split('\n')[0].split(' ')[1]
        self.brew_version = version
        return self.brew_version

    def _brew_cask_command_is_deprecated(self):
        return LooseVersion(self._get_brew_version()) >= LooseVersion('2.6.0')

    def _run(self):
        if self.upgrade_all:
            return self._upgrade_all()
        if self.casks:
            if self.state == 'installed':
                return self._install_casks()
            elif self.state == 'upgraded':
                return self._upgrade_casks()
            elif self.state == 'absent':
                return self._uninstall_casks()
        self.failed = True
        self.message = 'You must select a cask to install.'
        raise HomebrewCaskException(self.message)

    def _run_command_with_sudo_password(self, cmd):
        rc, out, err = ('', '', '')
        with tempfile.NamedTemporaryFile() as sudo_askpass_file:
            sudo_askpass_file.write(b"#!/bin/sh\n\necho '%s'\n" % to_bytes(self.sudo_password))
            os.chmod(sudo_askpass_file.name, 448)
            sudo_askpass_file.file.close()
            rc, out, err = self.module.run_command(cmd, environ_update={'SUDO_ASKPASS': sudo_askpass_file.name})
            self.module.add_cleanup_file(sudo_askpass_file.name)
        return (rc, out, err)

    def _update_homebrew(self):
        rc, out, err = self.module.run_command([self.brew_path, 'update'])
        if rc == 0:
            if out and isinstance(out, string_types):
                already_updated = any((re.search('Already up-to-date.', s.strip(), re.IGNORECASE) for s in out.split('\n') if s))
                if not already_updated:
                    self.changed = True
                    self.message = 'Homebrew updated successfully.'
                else:
                    self.message = 'Homebrew already up-to-date.'
            return True
        else:
            self.failed = True
            self.message = err.strip()
            raise HomebrewCaskException(self.message)

    def _upgrade_all(self):
        if self.module.check_mode:
            self.changed = True
            self.message = 'Casks would be upgraded.'
            raise HomebrewCaskException(self.message)
        if self._brew_cask_command_is_deprecated():
            cmd = [self.brew_path, 'upgrade', '--cask']
        else:
            cmd = [self.brew_path, 'cask', 'upgrade']
        if self.greedy:
            cmd = cmd + ['--greedy']
        rc, out, err = ('', '', '')
        if self.sudo_password:
            rc, out, err = self._run_command_with_sudo_password(cmd)
        else:
            rc, out, err = self.module.run_command(cmd)
        if rc == 0:
            if re.search('==> No Casks to upgrade', out.strip(), re.IGNORECASE):
                self.message = 'Homebrew casks already upgraded.'
            else:
                self.changed = True
                self.message = 'Homebrew casks upgraded.'
            return True
        else:
            self.failed = True
            self.message = err.strip()
            raise HomebrewCaskException(self.message)

    def _install_current_cask(self):
        if not self.valid_cask(self.current_cask):
            self.failed = True
            self.message = 'Invalid cask: {0}.'.format(self.current_cask)
            raise HomebrewCaskException(self.message)
        if '--force' not in self.install_options and self._current_cask_is_installed():
            self.unchanged_count += 1
            self.message = 'Cask already installed: {0}'.format(self.current_cask)
            return True
        if self.module.check_mode:
            self.changed = True
            self.message = 'Cask would be installed: {0}'.format(self.current_cask)
            raise HomebrewCaskException(self.message)
        if self._brew_cask_command_is_deprecated():
            base_opts = [self.brew_path, 'install', '--cask']
        else:
            base_opts = [self.brew_path, 'cask', 'install']
        opts = base_opts + [self.current_cask] + self.install_options
        cmd = [opt for opt in opts if opt]
        rc, out, err = ('', '', '')
        if self.sudo_password:
            rc, out, err = self._run_command_with_sudo_password(cmd)
        else:
            rc, out, err = self.module.run_command(cmd)
        if self._current_cask_is_installed():
            self.changed_count += 1
            self.changed = True
            self.message = 'Cask installed: {0}'.format(self.current_cask)
            return True
        elif self.accept_external_apps and re.search('Error: It seems there is already an App at', err):
            self.unchanged_count += 1
            self.message = 'Cask already installed: {0}'.format(self.current_cask)
            return True
        else:
            self.failed = True
            self.message = err.strip()
            raise HomebrewCaskException(self.message)

    def _install_casks(self):
        for cask in self.casks:
            self.current_cask = cask
            self._install_current_cask()
        return True

    def _upgrade_current_cask(self):
        command = 'upgrade'
        if not self.valid_cask(self.current_cask):
            self.failed = True
            self.message = 'Invalid cask: {0}.'.format(self.current_cask)
            raise HomebrewCaskException(self.message)
        if not self._current_cask_is_installed():
            command = 'install'
        if self._current_cask_is_installed() and (not self._current_cask_is_outdated()):
            self.message = 'Cask is already upgraded: {0}'.format(self.current_cask)
            self.unchanged_count += 1
            return True
        if self.module.check_mode:
            self.changed = True
            self.message = 'Cask would be upgraded: {0}'.format(self.current_cask)
            raise HomebrewCaskException(self.message)
        if self._brew_cask_command_is_deprecated():
            base_opts = [self.brew_path, command, '--cask']
        else:
            base_opts = [self.brew_path, 'cask', command]
        opts = base_opts + self.install_options + [self.current_cask]
        cmd = [opt for opt in opts if opt]
        rc, out, err = ('', '', '')
        if self.sudo_password:
            rc, out, err = self._run_command_with_sudo_password(cmd)
        else:
            rc, out, err = self.module.run_command(cmd)
        if self._current_cask_is_installed() and (not self._current_cask_is_outdated()):
            self.changed_count += 1
            self.changed = True
            self.message = 'Cask upgraded: {0}'.format(self.current_cask)
            return True
        else:
            self.failed = True
            self.message = err.strip()
            raise HomebrewCaskException(self.message)

    def _upgrade_casks(self):
        for cask in self.casks:
            self.current_cask = cask
            self._upgrade_current_cask()
        return True

    def _uninstall_current_cask(self):
        if not self.valid_cask(self.current_cask):
            self.failed = True
            self.message = 'Invalid cask: {0}.'.format(self.current_cask)
            raise HomebrewCaskException(self.message)
        if not self._current_cask_is_installed():
            self.unchanged_count += 1
            self.message = 'Cask already uninstalled: {0}'.format(self.current_cask)
            return True
        if self.module.check_mode:
            self.changed = True
            self.message = 'Cask would be uninstalled: {0}'.format(self.current_cask)
            raise HomebrewCaskException(self.message)
        if self._brew_cask_command_is_deprecated():
            base_opts = [self.brew_path, 'uninstall', '--cask']
        else:
            base_opts = [self.brew_path, 'cask', 'uninstall']
        opts = base_opts + [self.current_cask] + self.install_options
        cmd = [opt for opt in opts if opt]
        rc, out, err = ('', '', '')
        if self.sudo_password:
            rc, out, err = self._run_command_with_sudo_password(cmd)
        else:
            rc, out, err = self.module.run_command(cmd)
        if not self._current_cask_is_installed():
            self.changed_count += 1
            self.changed = True
            self.message = 'Cask uninstalled: {0}'.format(self.current_cask)
            return True
        else:
            self.failed = True
            self.message = err.strip()
            raise HomebrewCaskException(self.message)

    def _uninstall_casks(self):
        for cask in self.casks:
            self.current_cask = cask
            self._uninstall_current_cask()
        return True