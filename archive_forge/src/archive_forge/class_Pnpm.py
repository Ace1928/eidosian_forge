from __future__ import absolute_import, division, print_function
import json
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
class Pnpm(object):

    def __init__(self, module, **kwargs):
        self.module = module
        self.name = kwargs['name']
        self.alias = kwargs['alias']
        self.version = kwargs['version']
        self.path = kwargs['path']
        self.globally = kwargs['globally']
        self.executable = kwargs['executable']
        self.ignore_scripts = kwargs['ignore_scripts']
        self.no_optional = kwargs['no_optional']
        self.production = kwargs['production']
        self.dev = kwargs['dev']
        self.optional = kwargs['optional']
        self.alias_name_ver = None
        if self.alias is not None:
            self.alias_name_ver = self.alias + '@npm:'
        if self.name is not None:
            self.alias_name_ver = (self.alias_name_ver or '') + self.name
            if self.version is not None:
                self.alias_name_ver = self.alias_name_ver + '@' + str(self.version)
            else:
                self.alias_name_ver = self.alias_name_ver + '@latest'

    def _exec(self, args, run_in_check_mode=False, check_rc=True):
        if not self.module.check_mode or (self.module.check_mode and run_in_check_mode):
            cmd = self.executable + args
            if self.globally:
                cmd.append('-g')
            if self.ignore_scripts:
                cmd.append('--ignore-scripts')
            if self.no_optional:
                cmd.append('--no-optional')
            if self.production:
                cmd.append('-P')
            if self.dev:
                cmd.append('-D')
            if self.name and self.optional:
                cmd.append('-O')
            cwd = None
            if self.path:
                if not os.path.exists(self.path):
                    os.makedirs(self.path)
                if not os.path.isdir(self.path):
                    self.module.fail_json(msg='Path %s is not a directory' % self.path)
                if not self.alias_name_ver and (not os.path.isfile(os.path.join(self.path, 'package.json'))):
                    self.module.fail_json(msg='package.json does not exist in provided path')
                cwd = self.path
            _rc, out, err = self.module.run_command(cmd, check_rc=check_rc, cwd=cwd)
            return (out, err)
        return (None, None)

    def missing(self):
        if not os.path.isfile(os.path.join(self.path, 'pnpm-lock.yaml')):
            return True
        cmd = ['list', '--json']
        if self.name is not None:
            cmd.append(self.name)
        try:
            out, err = self._exec(cmd, True, False)
            if err is not None and err != '':
                raise Exception(out)
            data = json.loads(out)
        except Exception as e:
            self.module.fail_json(msg='Failed to parse pnpm output with error %s' % to_native(e))
        if 'error' in data:
            return True
        data = data[0]
        for typedep in ['dependencies', 'devDependencies', 'optionalDependencies', 'unsavedDependencies']:
            if typedep not in data:
                continue
            for dep, prop in data[typedep].items():
                if self.alias is not None and self.alias != dep:
                    continue
                name = prop['from'] if self.alias is not None else dep
                if self.name != name:
                    continue
                if self.version is None or self.version == prop['version']:
                    return False
                break
        return True

    def install(self):
        if self.alias_name_ver is not None:
            return self._exec(['add', self.alias_name_ver])
        return self._exec(['install'])

    def update(self):
        return self._exec(['update', '--latest'])

    def uninstall(self):
        if self.alias is not None:
            return self._exec(['remove', self.alias])
        return self._exec(['remove', self.name])

    def list_outdated(self):
        if not os.path.isfile(os.path.join(self.path, 'pnpm-lock.yaml')):
            return list()
        cmd = ['outdated', '--format', 'json']
        try:
            out, err = self._exec(cmd, True, False)
            if err is not None and err != '':
                raise Exception(out)
            data_lines = out.splitlines(True)
            out = None
            for line in data_lines:
                if len(line) > 0 and line[0] == '{':
                    out = line
                    continue
                if len(line) > 0 and line[0] == '}':
                    out += line
                    break
                if out is not None:
                    out += line
            data = json.loads(out)
        except Exception as e:
            self.module.fail_json(msg='Failed to parse pnpm output with error %s' % to_native(e))
        return data.keys()