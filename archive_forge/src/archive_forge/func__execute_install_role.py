from __future__ import (absolute_import, division, print_function)
from ansible.cli import CLI
import argparse
import functools
import json
import os.path
import pathlib
import re
import shutil
import sys
import textwrap
import time
import typing as t
from dataclasses import dataclass
from yaml.error import YAMLError
import ansible.constants as C
from ansible import context
from ansible.cli.arguments import option_helpers as opt_help
from ansible.errors import AnsibleError, AnsibleOptionsError
from ansible.galaxy import Galaxy, get_collections_galaxy_meta_info
from ansible.galaxy.api import GalaxyAPI, GalaxyError
from ansible.galaxy.collection import (
from ansible.galaxy.collection.concrete_artifact_manager import (
from ansible.galaxy.collection.gpg import GPG_ERROR_MAP
from ansible.galaxy.dependency_resolution.dataclasses import Requirement
from ansible.galaxy.role import GalaxyRole
from ansible.galaxy.token import BasicAuthToken, GalaxyToken, KeycloakToken, NoTokenSentinel
from ansible.module_utils.ansible_release import __version__ as ansible_version
from ansible.module_utils.common.collections import is_iterable
from ansible.module_utils.common.yaml import yaml_dump, yaml_load
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils import six
from ansible.parsing.dataloader import DataLoader
from ansible.parsing.yaml.loader import AnsibleLoader
from ansible.playbook.role.requirement import RoleRequirement
from ansible.template import Templar
from ansible.utils.collection_loader import AnsibleCollectionConfig
from ansible.utils.display import Display
from ansible.utils.plugin_docs import get_versioned_doclink
def _execute_install_role(self, requirements):
    role_file = context.CLIARGS['requirements']
    no_deps = context.CLIARGS['no_deps']
    force_deps = context.CLIARGS['force_with_deps']
    force = context.CLIARGS['force'] or force_deps
    for role in requirements:
        if role_file and context.CLIARGS['args'] and (role.name not in context.CLIARGS['args']):
            display.vvv('Skipping role %s' % role.name)
            continue
        display.vvv('Processing role %s ' % role.name)
        if role.install_info is not None:
            if role.install_info['version'] != role.version or force:
                if force:
                    display.display('- changing role %s from %s to %s' % (role.name, role.install_info['version'], role.version or 'unspecified'))
                    role.remove()
                else:
                    display.warning('- %s (%s) is already installed - use --force to change version to %s' % (role.name, role.install_info['version'], role.version or 'unspecified'))
                    continue
            elif not force:
                display.display('- %s is already installed, skipping.' % str(role))
                continue
        try:
            installed = role.install()
        except AnsibleError as e:
            display.warning(u'- %s was NOT installed successfully: %s ' % (role.name, to_text(e)))
            self.exit_without_ignore()
            continue
        if not no_deps and installed:
            if not role.metadata:
                display.warning('Meta file %s is empty. Skipping dependencies.' % role.path)
            else:
                role_dependencies = role.metadata_dependencies + role.requirements
                for dep in role_dependencies:
                    display.debug('Installing dep %s' % dep)
                    dep_req = RoleRequirement()
                    dep_info = dep_req.role_yaml_parse(dep)
                    dep_role = GalaxyRole(self.galaxy, self.lazy_role_api, **dep_info)
                    if '.' not in dep_role.name and '.' not in dep_role.src and (dep_role.scm is None):
                        continue
                    if dep_role.install_info is None:
                        if dep_role not in requirements:
                            display.display('- adding dependency: %s' % to_text(dep_role))
                            requirements.append(dep_role)
                        else:
                            display.display('- dependency %s already pending installation.' % dep_role.name)
                    elif dep_role.install_info['version'] != dep_role.version:
                        if force_deps:
                            display.display('- changing dependent role %s from %s to %s' % (dep_role.name, dep_role.install_info['version'], dep_role.version or 'unspecified'))
                            dep_role.remove()
                            requirements.append(dep_role)
                        else:
                            display.warning('- dependency %s (%s) from role %s differs from already installed version (%s), skipping' % (to_text(dep_role), dep_role.version, role.name, dep_role.install_info['version']))
                    elif force_deps:
                        requirements.append(dep_role)
                    else:
                        display.display('- dependency %s is already installed, skipping.' % dep_role.name)
        if not installed:
            display.warning('- %s was NOT installed successfully.' % role.name)
            self.exit_without_ignore()
    return 0