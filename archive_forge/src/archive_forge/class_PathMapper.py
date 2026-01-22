from __future__ import annotations
import collections
import os
import re
import time
import typing as t
from ..target import (
from ..util import (
from .python import (
from .csharp import (
from .powershell import (
from ..config import (
from ..metadata import (
from ..data import (
class PathMapper:
    """Map file paths to test commands and targets."""

    def __init__(self, args: TestConfig) -> None:
        self.args = args
        self.integration_all_target = get_integration_all_target(self.args)
        self.integration_targets = list(walk_integration_targets())
        self.module_targets = list(walk_module_targets())
        self.compile_targets = list(walk_compile_targets())
        self.units_targets = list(walk_units_targets())
        self.sanity_targets = list(walk_sanity_targets())
        self.powershell_targets = [target for target in self.sanity_targets if os.path.splitext(target.path)[1] in ('.ps1', '.psm1')]
        self.csharp_targets = [target for target in self.sanity_targets if os.path.splitext(target.path)[1] == '.cs']
        self.units_modules = set((target.module for target in self.units_targets if target.module))
        self.units_paths = set((a for target in self.units_targets for a in target.aliases))
        self.sanity_paths = set((target.path for target in self.sanity_targets))
        self.module_names_by_path = dict(((target.path, target.module) for target in self.module_targets))
        self.integration_targets_by_name = dict(((target.name, target) for target in self.integration_targets))
        self.integration_targets_by_alias = dict(((a, target) for target in self.integration_targets for a in target.aliases))
        self.posix_integration_by_module = dict(((m, target.name) for target in self.integration_targets if 'posix/' in target.aliases for m in target.modules))
        self.windows_integration_by_module = dict(((m, target.name) for target in self.integration_targets if 'windows/' in target.aliases for m in target.modules))
        self.network_integration_by_module = dict(((m, target.name) for target in self.integration_targets if 'network/' in target.aliases for m in target.modules))
        self.prefixes = load_integration_prefixes()
        self.integration_dependencies = analyze_integration_target_dependencies(self.integration_targets)
        self.python_module_utils_imports: dict[str, set[str]] = {}
        self.powershell_module_utils_imports: dict[str, set[str]] = {}
        self.csharp_module_utils_imports: dict[str, set[str]] = {}
        self.paths_to_dependent_targets: dict[str, set[IntegrationTarget]] = {}
        for target in self.integration_targets:
            for path in target.needs_file:
                if path not in self.paths_to_dependent_targets:
                    self.paths_to_dependent_targets[path] = set()
                self.paths_to_dependent_targets[path].add(target)

    def get_dependent_paths(self, path: str) -> list[str]:
        """Return a list of paths which depend on the given path, recursively expanding dependent paths as well."""
        unprocessed_paths = set(self.get_dependent_paths_non_recursive(path))
        paths = set()
        while unprocessed_paths:
            queued_paths = list(unprocessed_paths)
            paths |= unprocessed_paths
            unprocessed_paths = set()
            for queued_path in queued_paths:
                new_paths = self.get_dependent_paths_non_recursive(queued_path)
                for new_path in new_paths:
                    if new_path not in paths:
                        unprocessed_paths.add(new_path)
        return sorted(paths)

    def get_dependent_paths_non_recursive(self, path: str) -> list[str]:
        """Return a list of paths which depend on the given path, including dependent integration test target paths."""
        paths = self.get_dependent_paths_internal(path)
        paths += [target.path + '/' for target in self.paths_to_dependent_targets.get(path, set())]
        paths = sorted(set(paths))
        return paths

    def get_dependent_paths_internal(self, path: str) -> list[str]:
        """Return a list of paths which depend on the given path."""
        ext = os.path.splitext(os.path.split(path)[1])[1]
        if is_subdir(path, data_context().content.module_utils_path):
            if ext == '.py':
                return self.get_python_module_utils_usage(path)
            if ext == '.psm1':
                return self.get_powershell_module_utils_usage(path)
            if ext == '.cs':
                return self.get_csharp_module_utils_usage(path)
        if is_subdir(path, data_context().content.integration_targets_path):
            return self.get_integration_target_usage(path)
        return []

    def get_python_module_utils_usage(self, path: str) -> list[str]:
        """Return a list of paths which depend on the given path which is a Python module_utils file."""
        if not self.python_module_utils_imports:
            display.info('Analyzing python module_utils imports...')
            before = time.time()
            self.python_module_utils_imports = get_python_module_utils_imports(self.compile_targets)
            after = time.time()
            display.info('Processed %d python module_utils in %d second(s).' % (len(self.python_module_utils_imports), after - before))
        name = get_python_module_utils_name(path)
        return sorted(self.python_module_utils_imports[name])

    def get_powershell_module_utils_usage(self, path: str) -> list[str]:
        """Return a list of paths which depend on the given path which is a PowerShell module_utils file."""
        if not self.powershell_module_utils_imports:
            display.info('Analyzing powershell module_utils imports...')
            before = time.time()
            self.powershell_module_utils_imports = get_powershell_module_utils_imports(self.powershell_targets)
            after = time.time()
            display.info('Processed %d powershell module_utils in %d second(s).' % (len(self.powershell_module_utils_imports), after - before))
        name = get_powershell_module_utils_name(path)
        return sorted(self.powershell_module_utils_imports[name])

    def get_csharp_module_utils_usage(self, path: str) -> list[str]:
        """Return a list of paths which depend on the given path which is a C# module_utils file."""
        if not self.csharp_module_utils_imports:
            display.info('Analyzing C# module_utils imports...')
            before = time.time()
            self.csharp_module_utils_imports = get_csharp_module_utils_imports(self.powershell_targets, self.csharp_targets)
            after = time.time()
            display.info('Processed %d C# module_utils in %d second(s).' % (len(self.csharp_module_utils_imports), after - before))
        name = get_csharp_module_utils_name(path)
        return sorted(self.csharp_module_utils_imports[name])

    def get_integration_target_usage(self, path: str) -> list[str]:
        """Return a list of paths which depend on the given path which is an integration target file."""
        target_name = path.split('/')[3]
        dependents = [os.path.join(data_context().content.integration_targets_path, target) + os.path.sep for target in sorted(self.integration_dependencies.get(target_name, set()))]
        return dependents

    def classify(self, path: str) -> t.Optional[dict[str, str]]:
        """Classify the given path and return an optional dictionary of the results."""
        result = self._classify(path)
        if result is None:
            return None
        if path in self.sanity_paths and 'sanity' not in result:
            result['sanity'] = path
        return result

    def _classify(self, path: str) -> t.Optional[dict[str, str]]:
        """Return the classification for the given path."""
        if data_context().content.is_ansible:
            return self._classify_ansible(path)
        if data_context().content.collection:
            return self._classify_collection(path)
        return None

    def _classify_common(self, path: str) -> t.Optional[dict[str, str]]:
        """Return the classification for the given path using rules common to all layouts."""
        dirname = os.path.dirname(path)
        filename = os.path.basename(path)
        name, ext = os.path.splitext(filename)
        minimal: dict[str, str] = {}
        if os.path.sep not in path:
            if filename in ('azure-pipelines.yml',):
                return all_tests(self.args)
        if is_subdir(path, '.azure-pipelines'):
            return all_tests(self.args)
        if is_subdir(path, '.github'):
            return minimal
        if is_subdir(path, data_context().content.integration_targets_path):
            if not os.path.exists(path):
                return minimal
            target = self.integration_targets_by_name.get(path.split('/')[3])
            if not target:
                display.warning('Unexpected non-target found: %s' % path)
                return minimal
            if 'hidden/' in target.aliases:
                return minimal
            return {'integration': target.name if 'posix/' in target.aliases else None, 'windows-integration': target.name if 'windows/' in target.aliases else None, 'network-integration': target.name if 'network/' in target.aliases else None, FOCUSED_TARGET: target.name}
        if is_subdir(path, data_context().content.integration_path):
            if dirname == data_context().content.integration_path:
                for command in ('integration', 'windows-integration', 'network-integration'):
                    if name == command and ext == '.cfg':
                        return {command: self.integration_all_target}
                    if name == command + '.requirements' and ext == '.txt':
                        return {command: self.integration_all_target}
            return {'integration': self.integration_all_target, 'windows-integration': self.integration_all_target, 'network-integration': self.integration_all_target}
        if is_subdir(path, data_context().content.sanity_path):
            return {'sanity': 'all'}
        if is_subdir(path, data_context().content.unit_path):
            if path in self.units_paths:
                return {'units': path}
            test_path = os.path.dirname(path)
            while test_path:
                if test_path + '/' in self.units_paths:
                    return {'units': test_path + '/'}
                test_path = os.path.dirname(test_path)
        if is_subdir(path, data_context().content.module_path):
            module_name = self.module_names_by_path.get(path)
            if module_name:
                return {'units': module_name if module_name in self.units_modules else None, 'integration': self.posix_integration_by_module.get(module_name) if ext == '.py' else None, 'windows-integration': self.windows_integration_by_module.get(module_name) if ext in ['.cs', '.ps1'] else None, 'network-integration': self.network_integration_by_module.get(module_name), FOCUSED_TARGET: module_name}
            return minimal
        if is_subdir(path, data_context().content.module_utils_path):
            if ext == '.cs':
                return minimal
            if ext == '.psm1':
                return minimal
            if ext == '.py':
                return minimal
        if is_subdir(path, data_context().content.plugin_paths['action']):
            if ext == '.py':
                if name.startswith('net_'):
                    network_target = 'network/.*_%s' % name[4:]
                    if any((re.search('^%s$' % network_target, alias) for alias in self.integration_targets_by_alias)):
                        return {'network-integration': network_target, 'units': 'all'}
                    return {'network-integration': self.integration_all_target, 'units': 'all'}
                if self.prefixes.get(name) == 'network':
                    network_platform = name
                elif name.endswith('_config') and self.prefixes.get(name[:-7]) == 'network':
                    network_platform = name[:-7]
                elif name.endswith('_template') and self.prefixes.get(name[:-9]) == 'network':
                    network_platform = name[:-9]
                else:
                    network_platform = None
                if network_platform:
                    network_target = 'network/%s/' % network_platform
                    if network_target in self.integration_targets_by_alias:
                        return {'network-integration': network_target, 'units': 'all'}
                    display.warning('Integration tests for "%s" not found.' % network_target, unique=True)
                    return {'units': 'all'}
        if is_subdir(path, data_context().content.plugin_paths['connection']):
            units_dir = os.path.join(data_context().content.unit_path, 'plugins', 'connection')
            if name == '__init__':
                return {'integration': self.integration_all_target, 'windows-integration': self.integration_all_target, 'network-integration': self.integration_all_target, 'units': os.path.join(units_dir, '')}
            units_path = os.path.join(units_dir, 'test_%s.py' % name)
            if units_path not in self.units_paths:
                units_path = None
            integration_name = 'connection_%s' % name
            if integration_name not in self.integration_targets_by_name:
                integration_name = None
            windows_integration_name = 'connection_windows_%s' % name
            if windows_integration_name not in self.integration_targets_by_name:
                windows_integration_name = None
            if name in ['winrm', 'psrp']:
                return {'windows-integration': self.integration_all_target, 'units': units_path}
            if name == 'local':
                return {'integration': self.integration_all_target, 'network-integration': self.integration_all_target, 'units': units_path}
            if name == 'network_cli':
                return {'network-integration': self.integration_all_target, 'units': units_path}
            if name == 'paramiko_ssh':
                return {'integration': integration_name, 'network-integration': self.integration_all_target, 'units': units_path}
            return {'integration': integration_name, 'windows-integration': windows_integration_name, 'units': units_path}
        if is_subdir(path, data_context().content.plugin_paths['doc_fragments']):
            return {'sanity': 'all'}
        if is_subdir(path, data_context().content.plugin_paths['inventory']):
            if name == '__init__':
                return all_tests(self.args)
            test_all = ['host_list', 'script', 'yaml', 'ini', 'auto']
            if name in test_all:
                posix_integration_fallback = get_integration_all_target(self.args)
            else:
                posix_integration_fallback = None
            target = self.integration_targets_by_name.get('inventory_%s' % name)
            units_dir = os.path.join(data_context().content.unit_path, 'plugins', 'inventory')
            units_path = os.path.join(units_dir, 'test_%s.py' % name)
            if units_path not in self.units_paths:
                units_path = None
            return {'integration': target.name if target and 'posix/' in target.aliases else posix_integration_fallback, 'windows-integration': target.name if target and 'windows/' in target.aliases else None, 'network-integration': target.name if target and 'network/' in target.aliases else None, 'units': units_path, FOCUSED_TARGET: target.name if target else None}
        if is_subdir(path, data_context().content.plugin_paths['filter']):
            return self._simple_plugin_tests('filter', name)
        if is_subdir(path, data_context().content.plugin_paths['lookup']):
            return self._simple_plugin_tests('lookup', name)
        if is_subdir(path, data_context().content.plugin_paths['terminal']) or is_subdir(path, data_context().content.plugin_paths['cliconf']) or is_subdir(path, data_context().content.plugin_paths['netconf']):
            if ext == '.py':
                if name in self.prefixes and self.prefixes[name] == 'network':
                    network_target = 'network/%s/' % name
                    if network_target in self.integration_targets_by_alias:
                        return {'network-integration': network_target, 'units': 'all'}
                    display.warning('Integration tests for "%s" not found.' % network_target, unique=True)
                    return {'units': 'all'}
                return {'network-integration': self.integration_all_target, 'units': 'all'}
        if is_subdir(path, data_context().content.plugin_paths['test']):
            return self._simple_plugin_tests('test', name)
        return None

    def _classify_collection(self, path: str) -> t.Optional[dict[str, str]]:
        """Return the classification for the given path using rules specific to collections."""
        result = self._classify_common(path)
        if result is not None:
            return result
        filename = os.path.basename(path)
        dummy, ext = os.path.splitext(filename)
        minimal: dict[str, str] = {}
        if path.startswith('changelogs/'):
            return minimal
        if path.startswith('docs/'):
            return minimal
        if '/' not in path:
            if path in ('.gitignore', 'COPYING', 'LICENSE', 'Makefile'):
                return minimal
            if ext in ('.in', '.md', '.rst', '.toml', '.txt'):
                return minimal
        return None

    def _classify_ansible(self, path: str) -> t.Optional[dict[str, str]]:
        """Return the classification for the given path using rules specific to Ansible."""
        dirname = os.path.dirname(path)
        filename = os.path.basename(path)
        name, ext = os.path.splitext(filename)
        minimal: dict[str, str] = {}
        packaging = {'integration': 'packaging/'}
        if path.startswith('test/units/compat/'):
            return {'units': 'test/units/'}
        if dirname == '.azure-pipelines/commands':
            test_map = {'cloud.sh': 'integration:cloud/', 'linux.sh': 'integration:all', 'network.sh': 'network-integration:all', 'remote.sh': 'integration:all', 'sanity.sh': 'sanity:all', 'units.sh': 'units:all', 'windows.sh': 'windows-integration:all'}
            test_match = test_map.get(filename)
            if test_match:
                test_command, test_target = test_match.split(':')
                return {test_command: test_target}
            cloud_target = f'cloud/{name}/'
            if cloud_target in self.integration_targets_by_alias:
                return {'integration': cloud_target}
        result = self._classify_common(path)
        if result is not None:
            return result
        if path.startswith('bin/'):
            return all_tests(self.args)
        if path.startswith('changelogs/'):
            return minimal
        if path.startswith('hacking/'):
            return minimal
        if path.startswith('lib/ansible/executor/powershell/'):
            units_path = 'test/units/executor/powershell/'
            if units_path not in self.units_paths:
                units_path = None
            return {'windows-integration': self.integration_all_target, 'units': units_path}
        if path.startswith('lib/ansible/'):
            return all_tests(self.args)
        if path.startswith('licenses/'):
            return minimal
        if path.startswith('packaging/'):
            packaging_target = f'packaging_{os.path.splitext(path.split(os.path.sep)[1])[0]}'
            if packaging_target in self.integration_targets_by_name:
                return {'integration': packaging_target}
            return minimal
        if path.startswith('test/ansible_test/'):
            return minimal
        if path.startswith('test/lib/ansible_test/config/'):
            if name.startswith('cloud-config-'):
                cloud_target = 'cloud/%s/' % name.split('-')[2].split('.')[0]
                if cloud_target in self.integration_targets_by_alias:
                    return {'integration': cloud_target}
        if path.startswith('test/lib/ansible_test/_data/completion/'):
            if path == 'test/lib/ansible_test/_data/completion/docker.txt':
                return all_tests(self.args, force=True)
        if path.startswith('test/lib/ansible_test/_internal/commands/integration/cloud/'):
            cloud_target = 'cloud/%s/' % name
            if cloud_target in self.integration_targets_by_alias:
                return {'integration': cloud_target}
            return all_tests(self.args)
        if path.startswith('test/lib/ansible_test/_internal/commands/sanity/'):
            return {'sanity': 'all', 'integration': 'ansible-test/'}
        if path.startswith('test/lib/ansible_test/_internal/commands/units/'):
            return {'units': 'all', 'integration': 'ansible-test/'}
        if path.startswith('test/lib/ansible_test/_data/requirements/'):
            if name in ('integration', 'network-integration', 'windows-integration'):
                return {name: self.integration_all_target}
            if name in ('sanity', 'units'):
                return {name: 'all'}
        if path.startswith('test/lib/ansible_test/_util/controller/sanity/') or path.startswith('test/lib/ansible_test/_util/target/sanity/'):
            return {'sanity': 'all', 'integration': 'ansible-test/'}
        if path.startswith('test/lib/ansible_test/_util/target/pytest/'):
            return {'units': 'all', 'integration': 'ansible-test/'}
        if path.startswith('test/lib/'):
            return all_tests(self.args)
        if path.startswith('test/support/'):
            return all_tests(self.args)
        if '/' not in path:
            if path in ('.gitattributes', '.gitignore', '.mailmap', 'COPYING', 'Makefile'):
                return minimal
            if path in ('MANIFEST.in', 'pyproject.toml', 'requirements.txt', 'setup.cfg', 'setup.py'):
                return packaging
            if ext in ('.md', '.rst'):
                return minimal
        return None

    def _simple_plugin_tests(self, plugin_type: str, plugin_name: str) -> dict[str, t.Optional[str]]:
        """
        Return tests for the given plugin type and plugin name.
        This function is useful for plugin types which do not require special processing.
        """
        if plugin_name == '__init__':
            return all_tests(self.args, True)
        integration_target = self.integration_targets_by_name.get('%s_%s' % (plugin_type, plugin_name))
        if integration_target:
            integration_name = integration_target.name
        else:
            integration_name = None
        units_path = os.path.join(data_context().content.unit_path, 'plugins', plugin_type, 'test_%s.py' % plugin_name)
        if units_path not in self.units_paths:
            units_path = None
        return dict(integration=integration_name, units=units_path)