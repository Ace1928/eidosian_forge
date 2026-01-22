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
def _build_summary(self, role, collection, argspec):
    """Build a summary dict for a role.

        Returns a simplified role arg spec containing only the role entry points and their
        short descriptions, and the role collection name (if applicable).

        :param role: The simple role name.
        :param collection: The collection containing the role (None or empty string if N/A).
        :param argspec: The complete role argspec data dict.

        :returns: A tuple with the FQCN role name and a summary dict.
        """
    if collection:
        fqcn = '.'.join([collection, role])
    else:
        fqcn = role
    summary = {}
    summary['collection'] = collection
    summary['entry_points'] = {}
    for ep in argspec.keys():
        entry_spec = argspec[ep] or {}
        summary['entry_points'][ep] = entry_spec.get('short_description', '')
    return (fqcn, summary)