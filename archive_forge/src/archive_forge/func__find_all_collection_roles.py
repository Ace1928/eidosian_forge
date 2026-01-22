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
def _find_all_collection_roles(self, name_filters=None, collection_filter=None):
    """Find all collection roles with an argument spec file.

        Note that argument specs do not actually need to exist within the spec file.

        :param name_filters: A tuple of one or more role names used to filter the results. These
            might be fully qualified with the collection name (e.g., community.general.roleA)
            or not (e.g., roleA).

        :param collection_filter: A list of strings containing the FQCN of a collection which will
            be used to limit results. This filter will take precedence over the name_filters.

        :returns: A set of tuples consisting of: role name, collection name, collection path
        """
    found = set()
    b_colldirs = list_collection_dirs(coll_filter=collection_filter)
    for b_path in b_colldirs:
        path = to_text(b_path, errors='surrogate_or_strict')
        collname = _get_collection_name_from_path(b_path)
        roles_dir = os.path.join(path, 'roles')
        if os.path.exists(roles_dir):
            for entry in os.listdir(roles_dir):
                for specfile in self.ROLE_ARGSPEC_FILES:
                    full_path = os.path.join(roles_dir, entry, 'meta', specfile)
                    if os.path.exists(full_path):
                        if name_filters is None:
                            found.add((entry, collname, path))
                        else:
                            for fqcn in name_filters:
                                if len(fqcn.split('.')) == 3:
                                    ns, col, role = fqcn.split('.')
                                    if '.'.join([ns, col]) == collname and entry == role:
                                        found.add((entry, collname, path))
                                elif fqcn == entry:
                                    found.add((entry, collname, path))
                        break
    return found