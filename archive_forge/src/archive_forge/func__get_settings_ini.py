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
def _get_settings_ini(self, settings, seen):
    sections = {}
    for o in sorted(settings.keys()):
        opt = settings[o]
        if not isinstance(opt, Mapping):
            continue
        if not opt.get('description'):
            new_sections = self._get_settings_ini(opt, seen)
            for s in new_sections:
                if s in sections:
                    sections[s].extend(new_sections[s])
                else:
                    sections[s] = new_sections[s]
            continue
        if isinstance(opt['description'], string_types):
            desc = '# (%s) %s' % (opt.get('type', 'string'), opt['description'])
        else:
            desc = '# (%s) ' % opt.get('type', 'string')
            desc += '\n# '.join(opt['description'])
        if 'ini' in opt and opt['ini']:
            entry = opt['ini'][-1]
            if entry['section'] not in seen:
                seen[entry['section']] = []
            if entry['section'] not in sections:
                sections[entry['section']] = []
            if entry['key'] not in seen[entry['section']]:
                seen[entry['section']].append(entry['key'])
                default = self.config.template_default(opt.get('default', ''), get_constants())
                if opt.get('type', '') == 'list' and (not isinstance(default, string_types)):
                    default = ', '.join(default)
                elif default is None:
                    default = ''
                if context.CLIARGS['commented']:
                    entry['key'] = ';%s' % entry['key']
                key = desc + '\n%s=%s' % (entry['key'], default)
                sections[entry['section']].append(key)
    return sections