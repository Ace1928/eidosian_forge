from __future__ import (absolute_import, division, print_function)
from ansible.cli import CLI
import os
import yaml
import shlex
import subprocess
from collections.abc import Mapping
from ansible import context
import ansible.plugins.loader as plugin_loader
from ansible import constants as C
from ansible.cli.arguments import option_helpers as opt_help
from ansible.config.manager import ConfigManager, Setting
from ansible.errors import AnsibleError, AnsibleOptionsError
from ansible.module_utils.common.text.converters import to_native, to_text, to_bytes
from ansible.module_utils.common.json import json_dump
from ansible.module_utils.six import string_types
from ansible.parsing.quoting import is_quoted
from ansible.parsing.yaml.dumper import AnsibleDumper
from ansible.utils.color import stringc
from ansible.utils.display import Display
from ansible.utils.path import unfrackpath
def _get_plugin_configs(self, ptype, plugins):
    loader = getattr(plugin_loader, '%s_loader' % ptype)
    output = []
    config_entries = {}
    if plugins:
        plugin_cs = []
        for plugin in plugins:
            p = loader.get(plugin, class_only=True)
            if p is None:
                display.warning('Skipping %s as we could not find matching plugin' % plugin)
            else:
                plugin_cs.append(loader.get(plugin, class_only=True))
    else:
        plugin_cs = loader.all(class_only=True)
    for plugin in plugin_cs:
        finalname = name = plugin._load_name
        if name.startswith('_'):
            if os.path.islink(plugin._original_path):
                continue
            finalname = name.replace('_', '', 1) + ' (DEPRECATED)'
        config_entries[finalname] = self.config.get_configuration_definitions(ptype, name)
        try:
            dump = loader.get(name, class_only=True)
        except Exception as e:
            display.warning('Skipping "%s" %s plugin, as we cannot load plugin to check config due to : %s' % (name, ptype, to_native(e)))
            continue
        for setting in config_entries[finalname].keys():
            try:
                v, o = C.config.get_config_value_and_origin(setting, cfile=self.config_file, plugin_type=ptype, plugin_name=name, variables=get_constants())
            except AnsibleError as e:
                if to_text(e).startswith('No setting was provided for required configuration'):
                    v = None
                    o = 'REQUIRED'
                else:
                    raise e
            if v is None and o is None:
                o = 'REQUIRED'
            config_entries[finalname][setting] = Setting(setting, v, o, None)
        results = self._render_settings(config_entries[finalname])
        if results:
            if context.CLIARGS['format'] == 'display':
                output.append('\n%s:\n%s' % (finalname, '_' * len(finalname)))
                output.extend(results)
            else:
                output.append({finalname: results})
    return output