from __future__ import absolute_import, division, print_function
import os
from ansible_collections.community.general.plugins.module_utils.cmd_runner import CmdRunner, cmd_runner_fmt
from ansible_collections.community.general.plugins.module_utils.module_helper import ModuleHelper
class CPANMinus(ModuleHelper):
    output_params = ['name', 'version']
    module = dict(argument_spec=dict(name=dict(type='str', aliases=['pkg']), version=dict(type='str'), from_path=dict(type='path'), notest=dict(type='bool', default=False), locallib=dict(type='path'), mirror=dict(type='str'), mirror_only=dict(type='bool', default=False), installdeps=dict(type='bool', default=False), executable=dict(type='path'), mode=dict(type='str', choices=['compatibility', 'new']), name_check=dict(type='str')), required_one_of=[('name', 'from_path')])
    command = 'cpanm'
    command_args_formats = dict(notest=cmd_runner_fmt.as_bool('--notest'), locallib=cmd_runner_fmt.as_opt_val('--local-lib'), mirror=cmd_runner_fmt.as_opt_val('--mirror'), mirror_only=cmd_runner_fmt.as_bool('--mirror-only'), installdeps=cmd_runner_fmt.as_bool('--installdeps'), pkg_spec=cmd_runner_fmt.as_list())

    def __init_module__(self):
        v = self.vars
        if v.mode is None:
            self.deprecate("The default value 'compatibility' for parameter 'mode' is being deprecated and it will be replaced by 'new'", version='9.0.0', collection_name='community.general')
            v.mode = 'compatibility'
        if v.mode == 'compatibility':
            if v.name_check:
                self.do_raise('Parameter name_check can only be used with mode=new')
        elif v.name and v.from_path:
            self.do_raise("Parameters 'name' and 'from_path' are mutually exclusive when 'mode=new'")
        self.command = v.executable if v.executable else self.command
        self.runner = CmdRunner(self.module, self.command, self.command_args_formats, check_rc=True)
        self.vars.binary = self.runner.binary

    def _is_package_installed(self, name, locallib, version):

        def process(rc, out, err):
            return rc == 0
        if name is None or name.endswith('.tar.gz'):
            return False
        version = '' if version is None else ' ' + version
        env = {'PERL5LIB': '%s/lib/perl5' % locallib} if locallib else {}
        runner = CmdRunner(self.module, ['perl', '-le'], {'mod': cmd_runner_fmt.as_list()}, check_rc=False, environ_update=env)
        with runner('mod', output_process=process) as ctx:
            return ctx.run(mod='use %s%s;' % (name, version))

    def sanitize_pkg_spec_version(self, pkg_spec, version):
        if version is None:
            return pkg_spec
        if pkg_spec.endswith('.tar.gz'):
            self.do_raise(msg="parameter 'version' must not be used when installing from a file")
        if os.path.isdir(pkg_spec):
            self.do_raise(msg="parameter 'version' must not be used when installing from a directory")
        if pkg_spec.endswith('.git'):
            if version.startswith('~'):
                self.do_raise(msg="operator '~' not allowed in version parameter when installing from git repository")
            version = version if version.startswith('@') else '@' + version
        elif version[0] not in ('@', '~'):
            version = '~' + version
        return pkg_spec + version

    def __run__(self):

        def process(rc, out, err):
            if self.vars.mode == 'compatibility' and rc != 0:
                self.do_raise(msg=err, cmd=self.vars.cmd_args)
            return 'is up to date' not in err and 'is up to date' not in out
        v = self.vars
        pkg_param = 'from_path' if v.from_path else 'name'
        if v.mode == 'compatibility':
            if self._is_package_installed(v.name, v.locallib, v.version):
                return
            pkg_spec = v[pkg_param]
        else:
            installed = self._is_package_installed(v.name_check, v.locallib, v.version) if v.name_check else False
            if installed:
                return
            pkg_spec = self.sanitize_pkg_spec_version(v[pkg_param], v.version)
        with self.runner(['notest', 'locallib', 'mirror', 'mirror_only', 'installdeps', 'pkg_spec'], output_process=process) as ctx:
            self.changed = ctx.run(pkg_spec=pkg_spec)