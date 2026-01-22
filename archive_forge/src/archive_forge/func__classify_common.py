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