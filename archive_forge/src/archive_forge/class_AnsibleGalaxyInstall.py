from __future__ import absolute_import, division, print_function
import re
from ansible_collections.community.general.plugins.module_utils.cmd_runner import CmdRunner, cmd_runner_fmt as fmt
from ansible_collections.community.general.plugins.module_utils.module_helper import ModuleHelper, ModuleHelperException
class AnsibleGalaxyInstall(ModuleHelper):
    _RE_GALAXY_VERSION = re.compile('^ansible-galaxy(?: \\[core)? (?P<version>\\d+\\.\\d+\\.\\d+)(?:\\.\\w+)?(?:\\])?')
    _RE_LIST_PATH = re.compile('^# (?P<path>.*)$')
    _RE_LIST_COLL = re.compile('^(?P<elem>\\w+\\.\\w+)\\s+(?P<version>[\\d\\.]+)\\s*$')
    _RE_LIST_ROLE = re.compile('^- (?P<elem>\\w+\\.\\w+),\\s+(?P<version>[\\d\\.]+)\\s*$')
    _RE_INSTALL_OUTPUT = None
    ansible_version = None
    output_params = ('type', 'name', 'dest', 'requirements_file', 'force', 'no_deps')
    module = dict(argument_spec=dict(type=dict(type='str', choices=('collection', 'role', 'both'), required=True), name=dict(type='str'), requirements_file=dict(type='path'), dest=dict(type='path'), force=dict(type='bool', default=False), no_deps=dict(type='bool', default=False), ack_ansible29=dict(type='bool', default=False, removed_in_version='9.0.0', removed_from_collection='community.general'), ack_min_ansiblecore211=dict(type='bool', default=False, removed_in_version='9.0.0', removed_from_collection='community.general')), mutually_exclusive=[('name', 'requirements_file')], required_one_of=[('name', 'requirements_file')], required_if=[('type', 'both', ['requirements_file'])], supports_check_mode=False)
    command = 'ansible-galaxy'
    command_args_formats = dict(type=fmt.as_func(lambda v: [] if v == 'both' else [v]), galaxy_cmd=fmt.as_list(), requirements_file=fmt.as_opt_val('-r'), dest=fmt.as_opt_val('-p'), force=fmt.as_bool('--force'), no_deps=fmt.as_bool('--no-deps'), version=fmt.as_bool('--version'), name=fmt.as_list())

    def _make_runner(self, lang):
        return CmdRunner(self.module, command=self.command, arg_formats=self.command_args_formats, force_lang=lang, check_rc=True)

    def _get_ansible_galaxy_version(self):

        class UnsupportedLocale(ModuleHelperException):
            pass

        def process(rc, out, err):
            if rc != 0 and 'unsupported locale setting' in err or (rc == 0 and 'cannot change locale' in err):
                raise UnsupportedLocale(msg=err)
            line = out.splitlines()[0]
            match = self._RE_GALAXY_VERSION.match(line)
            if not match:
                self.do_raise('Unable to determine ansible-galaxy version from: {0}'.format(line))
            version = match.group('version')
            version = tuple((int(x) for x in version.split('.')[:3]))
            return version
        try:
            runner = self._make_runner('C.UTF-8')
            with runner('version', check_rc=False, output_process=process) as ctx:
                return (runner, ctx.run(version=True))
        except UnsupportedLocale as e:
            runner = self._make_runner('en_US.UTF-8')
            with runner('version', check_rc=True, output_process=process) as ctx:
                return (runner, ctx.run(version=True))

    def __init_module__(self):
        self.runner, self.ansible_version = self._get_ansible_galaxy_version()
        if self.ansible_version < (2, 11):
            self.module.fail_json(msg='Support for Ansible 2.9 and ansible-base 2.10 has ben removed.')
        self._RE_INSTALL_OUTPUT = re.compile('^(?:(?P<collection>\\w+\\.\\w+)(?: \\(|:)(?P<cversion>[\\d\\.]+)\\)?|- (?P<role>\\w+\\.\\w+) \\((?P<rversion>[\\d\\.]+)\\)) was installed successfully$')
        self.vars.set('new_collections', {}, change=True)
        self.vars.set('new_roles', {}, change=True)
        if self.vars.type != 'collection':
            self.vars.installed_roles = self._list_roles()
        if self.vars.type != 'roles':
            self.vars.installed_collections = self._list_collections()

    def _list_element(self, _type, path_re, elem_re):

        def process(rc, out, err):
            return [] if 'None of the provided paths were usable' in out else out.splitlines()
        with self.runner('type galaxy_cmd dest', output_process=process, check_rc=False) as ctx:
            elems = ctx.run(type=_type, galaxy_cmd='list')
        elems_dict = {}
        current_path = None
        for line in elems:
            if line.startswith('#'):
                match = path_re.match(line)
                if not match:
                    continue
                if self.vars.dest is not None and match.group('path') != self.vars.dest:
                    current_path = None
                    continue
                current_path = match.group('path') if match else None
                elems_dict[current_path] = {}
            elif current_path is not None:
                match = elem_re.match(line)
                if not match or (self.vars.name is not None and match.group('elem') != self.vars.name):
                    continue
                elems_dict[current_path][match.group('elem')] = match.group('version')
        return elems_dict

    def _list_collections(self):
        return self._list_element('collection', self._RE_LIST_PATH, self._RE_LIST_COLL)

    def _list_roles(self):
        return self._list_element('role', self._RE_LIST_PATH, self._RE_LIST_ROLE)

    def __run__(self):

        def process(rc, out, err):
            for line in out.splitlines():
                match = self._RE_INSTALL_OUTPUT.match(line)
                if not match:
                    continue
                if match.group('collection'):
                    self.vars.new_collections[match.group('collection')] = match.group('cversion')
                elif match.group('role'):
                    self.vars.new_roles[match.group('role')] = match.group('rversion')
        with self.runner('type galaxy_cmd force no_deps dest requirements_file name', output_process=process) as ctx:
            ctx.run(galaxy_cmd='install')
            if self.verbosity > 2:
                self.vars.set('run_info', ctx.run_info)