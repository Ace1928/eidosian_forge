from __future__ import (absolute_import, division, print_function)
from abc import ABC
import types
import typing as t
from ansible import constants as C
from ansible.errors import AnsibleError
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six import string_types
from ansible.utils.display import Display
class AnsiblePlugin(ABC):
    allow_extras = False
    _load_name: str

    def __init__(self):
        self._options = {}
        self._defs = None

    def matches_name(self, possible_names):
        possible_fqcns = set()
        for name in possible_names:
            if '.' not in name:
                possible_fqcns.add(f'ansible.builtin.{name}')
            elif name.startswith('ansible.legacy.'):
                possible_fqcns.add(name.removeprefix('ansible.legacy.'))
            possible_fqcns.add(name)
        return bool(possible_fqcns.intersection(set(self.ansible_aliases)))

    def get_option_and_origin(self, option, hostvars=None):
        try:
            option_value, origin = C.config.get_config_value_and_origin(option, plugin_type=self.plugin_type, plugin_name=self._load_name, variables=hostvars)
        except AnsibleError as e:
            raise KeyError(to_native(e))
        return (option_value, origin)

    def get_option(self, option, hostvars=None):
        if option not in self._options:
            option_value, dummy = self.get_option_and_origin(option, hostvars=hostvars)
            self.set_option(option, option_value)
        return self._options.get(option)

    def get_options(self, hostvars=None):
        options = {}
        for option in self.option_definitions.keys():
            options[option] = self.get_option(option, hostvars=hostvars)
        return options

    def set_option(self, option, value):
        self._options[option] = value

    def set_options(self, task_keys=None, var_options=None, direct=None):
        """
        Sets the _options attribute with the configuration/keyword information for this plugin

        :arg task_keys: Dict with playbook keywords that affect this option
        :arg var_options: Dict with either 'connection variables'
        :arg direct: Dict with 'direct assignment'
        """
        self._options = C.config.get_plugin_options(self.plugin_type, self._load_name, keys=task_keys, variables=var_options, direct=direct)
        if self.allow_extras and var_options and ('_extras' in var_options):
            self.set_option('_extras', var_options['_extras'])

    def has_option(self, option):
        if not self._options:
            self.set_options()
        return option in self._options

    @property
    def plugin_type(self):
        return self.__class__.__name__.lower().replace('module', '')

    @property
    def option_definitions(self):
        if self._defs is None:
            self._defs = C.config.get_configuration_definitions(plugin_type=self.plugin_type, name=self._load_name)
        return self._defs

    def _check_required(self):
        pass