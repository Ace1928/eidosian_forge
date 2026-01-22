from __future__ import (absolute_import, division, print_function)
from ansible.cli import CLI
import pkgutil
import os
import os.path
import re
import textwrap
import traceback
import ansible.plugins.loader as plugin_loader
from pathlib import Path
from ansible import constants as C
from ansible import context
from ansible.cli.arguments import option_helpers as opt_help
from ansible.collections.list import list_collection_dirs
from ansible.errors import AnsibleError, AnsibleOptionsError, AnsibleParserError, AnsiblePluginNotFound
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.common.collections import is_sequence
from ansible.module_utils.common.json import json_dump
from ansible.module_utils.common.yaml import yaml_dump
from ansible.module_utils.compat import importlib
from ansible.module_utils.six import string_types
from ansible.parsing.plugin_docs import read_docstub
from ansible.parsing.utils.yaml import from_yaml
from ansible.parsing.yaml.dumper import AnsibleDumper
from ansible.plugins.list import list_plugins
from ansible.plugins.loader import action_loader, fragment_loader
from ansible.utils.collection_loader import AnsibleCollectionConfig, AnsibleCollectionRef
from ansible.utils.collection_loader._collection_finder import _get_collection_name_from_path
from ansible.utils.display import Display
from ansible.utils.plugin_docs import get_plugin_docs, get_docstring, get_versioned_doclink
def _get_roles_path(self):
    """
         Add any 'roles' subdir in playbook dir to the roles search path.
         And as a last resort, add the playbook dir itself. Order being:
           - 'roles' subdir of playbook dir
           - DEFAULT_ROLES_PATH (default in cliargs)
           - playbook dir (basedir)
         NOTE: This matches logic in RoleDefinition._load_role_path() method.
        """
    roles_path = context.CLIARGS['roles_path']
    if context.CLIARGS['basedir'] is not None:
        subdir = os.path.join(context.CLIARGS['basedir'], 'roles')
        if os.path.isdir(subdir):
            roles_path = (subdir,) + roles_path
        roles_path = roles_path + (context.CLIARGS['basedir'],)
    return roles_path