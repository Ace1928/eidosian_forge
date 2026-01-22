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