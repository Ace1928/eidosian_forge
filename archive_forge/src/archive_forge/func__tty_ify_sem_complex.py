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
@staticmethod
def _tty_ify_sem_complex(matcher):
    text = DocCLI._UNESCAPE.sub('\\1', matcher.group(1))
    value = None
    if '=' in text:
        text, value = text.split('=', 1)
    m = DocCLI._FQCN_TYPE_PREFIX_RE.match(text)
    if m:
        plugin_fqcn = m.group(1)
        plugin_type = m.group(2)
        text = m.group(3)
    elif text.startswith(DocCLI._IGNORE_MARKER):
        text = text[len(DocCLI._IGNORE_MARKER):]
        plugin_fqcn = plugin_type = ''
    else:
        plugin_fqcn = plugin_type = ''
    entrypoint = None
    if ':' in text:
        entrypoint, text = text.split(':', 1)
    if value is not None:
        text = f'{text}={value}'
    if plugin_fqcn and plugin_type:
        plugin_suffix = '' if plugin_type in ('role', 'module', 'playbook') else ' plugin'
        plugin = f'{plugin_type}{plugin_suffix} {plugin_fqcn}'
        if plugin_type == 'role' and entrypoint is not None:
            plugin = f'{plugin}, {entrypoint} entrypoint'
        return f"`{text}' (of {plugin})"
    return f"`{text}'"