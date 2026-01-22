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
def _get_settings_vars(self, settings, subkey):
    data = []
    if context.CLIARGS['commented']:
        prefix = '#'
    else:
        prefix = ''
    for setting in settings:
        if not settings[setting].get('description'):
            continue
        default = self.config.template_default(settings[setting].get('default', ''), get_constants())
        if subkey == 'env':
            stype = settings[setting].get('type', '')
            if stype == 'boolean':
                if default:
                    default = '1'
                else:
                    default = '0'
            elif default:
                if stype == 'list':
                    if not isinstance(default, string_types):
                        try:
                            default = ', '.join(default)
                        except Exception as e:
                            default = '%s' % to_native(default)
                if isinstance(default, string_types) and (not is_quoted(default)):
                    default = shlex.quote(default)
            elif default is None:
                default = ''
        if subkey in settings[setting] and settings[setting][subkey]:
            entry = settings[setting][subkey][-1]['name']
            if isinstance(settings[setting]['description'], string_types):
                desc = settings[setting]['description']
            else:
                desc = '\n#'.join(settings[setting]['description'])
            name = settings[setting].get('name', setting)
            data.append('# %s(%s): %s' % (name, settings[setting].get('type', 'string'), desc))
            if subkey == 'env':
                if entry.startswith('_ANSIBLE_'):
                    continue
                data.append('%s%s=%s' % (prefix, entry, default))
            elif subkey == 'vars':
                if entry.startswith('_ansible_'):
                    continue
                data.append(prefix + '%s: %s' % (entry, to_text(yaml_short(default), errors='surrogate_or_strict')))
            data.append('')
    return data