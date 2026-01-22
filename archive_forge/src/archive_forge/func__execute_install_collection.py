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
def _execute_install_collection(self, requirements, path, artifacts_manager):
    force = context.CLIARGS['force']
    ignore_errors = context.CLIARGS['ignore_errors']
    no_deps = context.CLIARGS['no_deps']
    force_with_deps = context.CLIARGS['force_with_deps']
    try:
        disable_gpg_verify = context.CLIARGS['disable_gpg_verify']
    except KeyError:
        if self._implicit_role:
            raise AnsibleError('Unable to properly parse command line arguments. Please use "ansible-galaxy collection install" instead of "ansible-galaxy install".')
        raise
    allow_pre_release = context.CLIARGS.get('allow_pre_release', False)
    upgrade = context.CLIARGS.get('upgrade', False)
    collections_path = C.COLLECTIONS_PATHS
    managed_paths = set((validate_collection_path(p) for p in C.COLLECTIONS_PATHS))
    read_req_paths = set((validate_collection_path(p) for p in AnsibleCollectionConfig.collection_paths))
    unexpected_path = C.GALAXY_COLLECTIONS_PATH_WARNING and (not any((p.startswith(path) for p in managed_paths)))
    if unexpected_path and any((p.startswith(path) for p in read_req_paths)):
        display.warning(f"The specified collections path '{path}' appears to be part of the pip Ansible package. Managing these directly with ansible-galaxy could break the Ansible package. Install collections to a configured collections path, which will take precedence over collections found in the PYTHONPATH.")
    elif unexpected_path:
        display.warning("The specified collections path '%s' is not part of the configured Ansible collections paths '%s'. The installed collection will not be picked up in an Ansible run, unless within a playbook-adjacent collections directory." % (to_text(path), to_text(':'.join(collections_path))))
    output_path = validate_collection_path(path)
    b_output_path = to_bytes(output_path, errors='surrogate_or_strict')
    if not os.path.exists(b_output_path):
        os.makedirs(b_output_path)
    install_collections(requirements, output_path, self.api_servers, ignore_errors, no_deps, force, force_with_deps, upgrade, allow_pre_release=allow_pre_release, artifacts_manager=artifacts_manager, disable_gpg_verify=disable_gpg_verify, offline=context.CLIARGS.get('offline', False), read_requirement_paths=read_req_paths)
    return 0