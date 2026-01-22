from __future__ import (absolute_import, division, print_function)
import os
import sys
from collections import defaultdict
from collections.abc import Mapping, MutableMapping, Sequence
from hashlib import sha1
from jinja2.exceptions import UndefinedError
from ansible import constants as C
from ansible.errors import AnsibleError, AnsibleParserError, AnsibleUndefinedVariable, AnsibleFileNotFound, AnsibleAssertionError, AnsibleTemplateError
from ansible.inventory.host import Host
from ansible.inventory.helpers import sort_groups, get_group_vars
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.six import text_type, string_types
from ansible.plugins.loader import lookup_loader
from ansible.vars.fact_cache import FactCache
from ansible.template import Templar
from ansible.utils.display import Display
from ansible.utils.listify import listify_lookup_plugin_terms
from ansible.utils.vars import combine_vars, load_extra_vars, load_options_vars
from ansible.utils.unsafe_proxy import wrap_var
from ansible.vars.clean import namespace_facts, clean_facts
from ansible.vars.plugins import get_vars_from_inventory_sources, get_vars_from_path
class VarsWithSources(MutableMapping):
    """
    Dict-like class for vars that also provides source information for each var

    This class can only store the source for top-level vars. It does no tracking
    on its own, just shows a debug message with the information that it is provided
    when a particular var is accessed.
    """

    def __init__(self, *args, **kwargs):
        """ Dict-compatible constructor """
        self.data = dict(*args, **kwargs)
        self.sources = {}

    @classmethod
    def new_vars_with_sources(cls, data, sources):
        """ Alternate constructor method to instantiate class with sources """
        v = cls(data)
        v.sources = sources
        return v

    def get_source(self, key):
        return self.sources.get(key, None)

    def __getitem__(self, key):
        val = self.data[key]
        display.debug("variable '%s' from source: %s" % (key, self.sources.get(key, 'unknown')))
        return val

    def __setitem__(self, key, value):
        self.data[key] = value

    def __delitem__(self, key):
        del self.data[key]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __contains__(self, key):
        return self.data.__contains__(key)

    def copy(self):
        return VarsWithSources.new_vars_with_sources(self.data.copy(), self.sources.copy())