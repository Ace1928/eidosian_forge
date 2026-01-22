from __future__ import (absolute_import, division, print_function)
import locale
import os
import sys
from importlib.metadata import version
from ansible.module_utils.compat.version import LooseVersion
import errno
import getpass
import subprocess
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from ansible import context
from ansible.cli.arguments import option_helpers as opt_help
from ansible.errors import AnsibleError, AnsibleOptionsError, AnsibleParserError
from ansible.inventory.manager import InventoryManager
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.module_utils.common.collections import is_sequence
from ansible.module_utils.common.file import is_executable
from ansible.parsing.dataloader import DataLoader
from ansible.parsing.vault import PromptVaultSecret, get_file_vault_secret
from ansible.plugins.loader import add_all_plugin_dirs, init_plugin_loader
from ansible.release import __version__
from ansible.utils.collection_loader import AnsibleCollectionConfig
from ansible.utils.collection_loader._collection_finder import _get_collection_name_from_path
from ansible.utils.path import unfrackpath
from ansible.utils.unsafe_proxy import to_unsafe_text
from ansible.vars.manager import VariableManager
@staticmethod
def _play_prereqs():
    options = context.CLIARGS
    loader = DataLoader()
    basedir = options.get('basedir', False)
    if basedir:
        loader.set_basedir(basedir)
        add_all_plugin_dirs(basedir)
        AnsibleCollectionConfig.playbook_paths = basedir
        default_collection = _get_collection_name_from_path(basedir)
        if default_collection:
            display.warning(u'running with default collection {0}'.format(default_collection))
            AnsibleCollectionConfig.default_collection = default_collection
    vault_ids = list(options['vault_ids'])
    default_vault_ids = C.DEFAULT_VAULT_IDENTITY_LIST
    vault_ids = default_vault_ids + vault_ids
    vault_secrets = CLI.setup_vault_secrets(loader, vault_ids=vault_ids, vault_password_files=list(options['vault_password_files']), ask_vault_pass=options['ask_vault_pass'], auto_prompt=False)
    loader.set_vault_secrets(vault_secrets)
    inventory = InventoryManager(loader=loader, sources=options['inventory'], cache=not options.get('flush_cache'))
    variable_manager = VariableManager(loader=loader, inventory=inventory, version_info=CLI.version_info(gitinfo=False))
    return (loader, inventory, variable_manager)