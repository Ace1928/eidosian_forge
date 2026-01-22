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
def _get_keywords_docs(keys):
    data = {}
    descs = DocCLI._list_keywords()
    for key in keys:
        if key.startswith('with_'):
            keyword = 'loop'
        elif key == 'async':
            keyword = 'async_val'
        else:
            keyword = key
        try:
            kdata = {'description': descs[key]}
            kdata['applies_to'] = []
            for pobj in PB_OBJECTS:
                if pobj not in PB_LOADED:
                    obj_class = 'ansible.playbook.%s' % pobj.lower()
                    loaded_class = importlib.import_module(obj_class)
                    PB_LOADED[pobj] = getattr(loaded_class, pobj, None)
                if keyword in PB_LOADED[pobj].fattributes:
                    kdata['applies_to'].append(pobj)
                    if 'type' not in kdata:
                        fa = PB_LOADED[pobj].fattributes.get(keyword)
                        if getattr(fa, 'private'):
                            kdata = {}
                            raise KeyError
                        kdata['type'] = getattr(fa, 'isa', 'string')
                        if keyword.endswith('when') or keyword in ('until',):
                            kdata['template'] = 'implicit'
                        elif getattr(fa, 'static'):
                            kdata['template'] = 'static'
                        else:
                            kdata['template'] = 'explicit'
                        for visible in ('alias', 'priority'):
                            kdata[visible] = getattr(fa, visible)
            for k in list(kdata.keys()):
                if kdata[k] is None:
                    del kdata[k]
            data[key] = kdata
        except (AttributeError, KeyError) as e:
            display.warning("Skipping Invalid keyword '%s' specified: %s" % (key, to_text(e)))
            if display.verbosity >= 3:
                display.verbose(traceback.format_exc())
    return data