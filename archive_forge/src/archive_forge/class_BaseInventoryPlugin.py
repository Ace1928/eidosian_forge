from __future__ import (absolute_import, division, print_function)
import hashlib
import os
import string
from collections.abc import Mapping
from ansible.errors import AnsibleError, AnsibleParserError
from ansible.inventory.group import to_safe_group_name as original_safe
from ansible.parsing.utils.addresses import parse_address
from ansible.plugins import AnsiblePlugin
from ansible.plugins.cache import CachePluginAdjudicator as CacheObject
from ansible.module_utils.common.text.converters import to_bytes, to_native
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.six import string_types
from ansible.template import Templar
from ansible.utils.display import Display
from ansible.utils.vars import combine_vars, load_extra_vars
class BaseInventoryPlugin(AnsiblePlugin):
    """ Parses an Inventory Source"""
    TYPE = 'generator'
    _sanitize_group_name = staticmethod(to_safe_group_name)

    def __init__(self):
        super(BaseInventoryPlugin, self).__init__()
        self._options = {}
        self.inventory = None
        self.display = display
        self._vars = {}

    def parse(self, inventory, loader, path, cache=True):
        """ Populates inventory from the given data. Raises an error on any parse failure
            :arg inventory: a copy of the previously accumulated inventory data,
                 to be updated with any new data this plugin provides.
                 The inventory can be empty if no other source/plugin ran successfully.
            :arg loader: a reference to the DataLoader, which can read in YAML and JSON files,
                 it also has Vault support to automatically decrypt files.
            :arg path: the string that represents the 'inventory source',
                 normally a path to a configuration file for this inventory,
                 but it can also be a raw string for this plugin to consume
            :arg cache: a boolean that indicates if the plugin should use the cache or not
                 you can ignore if this plugin does not implement caching.
        """
        self.loader = loader
        self.inventory = inventory
        self.templar = Templar(loader=loader)
        self._vars = load_extra_vars(loader)

    def verify_file(self, path):
        """ Verify if file is usable by this plugin, base does minimal accessibility check
            :arg path: a string that was passed as an inventory source,
                 it normally is a path to a config file, but this is not a requirement,
                 it can also be parsed itself as the inventory data to process.
                 So only call this base class if you expect it to be a file.
        """
        valid = False
        b_path = to_bytes(path, errors='surrogate_or_strict')
        if os.path.exists(b_path) and os.access(b_path, os.R_OK):
            valid = True
        else:
            self.display.vvv('Skipping due to inventory source not existing or not being readable by the current user')
        return valid

    def _populate_host_vars(self, hosts, variables, group=None, port=None):
        if not isinstance(variables, Mapping):
            raise AnsibleParserError('Invalid data from file, expected dictionary and got:\n\n%s' % to_native(variables))
        for host in hosts:
            self.inventory.add_host(host, group=group, port=port)
            for k in variables:
                self.inventory.set_variable(host, k, variables[k])

    def _read_config_data(self, path):
        """ validate config and set options as appropriate
            :arg path: path to common yaml format config file for this plugin
        """
        config = {}
        try:
            config = self.loader.load_from_file(path, cache=False)
        except Exception as e:
            raise AnsibleParserError(to_native(e))
        valid_names = getattr(self, '_redirected_names') or [self.NAME]
        if not config:
            raise AnsibleParserError('%s is empty' % to_native(path))
        elif config.get('plugin') not in valid_names:
            raise AnsibleParserError('Incorrect plugin name in file: %s' % config.get('plugin', 'none found'))
        elif not isinstance(config, Mapping):
            raise AnsibleParserError('inventory source has invalid structure, it should be a dictionary, got: %s' % type(config))
        self.set_options(direct=config, var_options=self._vars)
        if 'cache' in self._options and self.get_option('cache'):
            cache_option_keys = [('_uri', 'cache_connection'), ('_timeout', 'cache_timeout'), ('_prefix', 'cache_prefix')]
            cache_options = dict(((opt[0], self.get_option(opt[1])) for opt in cache_option_keys if self.get_option(opt[1]) is not None))
            self._cache = get_cache_plugin(self.get_option('cache_plugin'), **cache_options)
        return config

    def _consume_options(self, data):
        """ update existing options from alternate configuration sources not normally used by Ansible.
            Many API libraries already have existing configuration sources, this allows plugin author to leverage them.
            :arg data: key/value pairs that correspond to configuration options for this plugin
        """
        for k in self._options:
            if k in data:
                self._options[k] = data.pop(k)

    def _expand_hostpattern(self, hostpattern):
        """
        Takes a single host pattern and returns a list of hostnames and an
        optional port number that applies to all of them.
        """
        try:
            pattern, port = parse_address(hostpattern, allow_ranges=True)
        except Exception:
            pattern = hostpattern
            port = None
        if detect_range(pattern):
            hostnames = expand_hostname_range(pattern)
        else:
            hostnames = [pattern]
        return (hostnames, port)