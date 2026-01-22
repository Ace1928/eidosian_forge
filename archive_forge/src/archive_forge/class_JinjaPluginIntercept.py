from __future__ import (absolute_import, division, print_function)
import ast
import datetime
import os
import pwd
import re
import time
from collections.abc import Iterator, Sequence, Mapping, MappingView, MutableMapping
from contextlib import contextmanager
from numbers import Number
from traceback import format_exc
from jinja2.exceptions import TemplateSyntaxError, UndefinedError, SecurityError
from jinja2.loaders import FileSystemLoader
from jinja2.nativetypes import NativeEnvironment
from jinja2.runtime import Context, StrictUndefined
from ansible import constants as C
from ansible.errors import (
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_native, to_text, to_bytes
from ansible.module_utils.common.collections import is_sequence
from ansible.plugins.loader import filter_loader, lookup_loader, test_loader
from ansible.template.native_helpers import ansible_native_concat, ansible_eval_concat, ansible_concat
from ansible.template.template import AnsibleJ2Template
from ansible.template.vars import AnsibleJ2Vars
from ansible.utils.display import Display
from ansible.utils.listify import listify_lookup_plugin_terms
from ansible.utils.native_jinja import NativeJinjaText
from ansible.utils.unsafe_proxy import to_unsafe_text, wrap_var, AnsibleUnsafeText, AnsibleUnsafeBytes, NativeJinjaUnsafeText
class JinjaPluginIntercept(MutableMapping):
    """ Simulated dict class that loads Jinja2Plugins at request
        otherwise all plugins would need to be loaded a priori.

        NOTE: plugin_loader still loads all 'builtin/legacy' at
        start so only collection plugins are really at request.
    """

    def __init__(self, delegatee, pluginloader, *args, **kwargs):
        super(JinjaPluginIntercept, self).__init__(*args, **kwargs)
        self._pluginloader = pluginloader
        self._delegatee = delegatee
        self._seen_it = set()

    def __getitem__(self, key):
        if not isinstance(key, string_types):
            raise ValueError('key must be a string, got %s instead' % type(key))
        original_exc = None
        if key not in self._seen_it:
            self._seen_it.add(key)
            plugin = None
            try:
                plugin = self._pluginloader.get(key)
            except (AnsibleError, KeyError) as e:
                original_exc = e
            except Exception as e:
                display.vvvv('Unexpected plugin load (%s) exception: %s' % (key, to_native(e)))
                raise e
            if plugin:
                self._delegatee[key] = plugin.j2_function
        try:
            func = self._delegatee[key]
        except KeyError as e:
            self._seen_it.remove(key)
            raise TemplateSyntaxError('Could not load "%s": %s' % (key, to_native(original_exc or e)), 0)
        if self._pluginloader.type == 'filter':
            if key in C.STRING_TYPE_FILTERS:
                func = _wrap_native_text(func)
            else:
                func = _unroll_iterator(func)
        return func

    def __setitem__(self, key, value):
        return self._delegatee.__setitem__(key, value)

    def __delitem__(self, key):
        raise NotImplementedError()

    def __iter__(self):
        return iter(self._delegatee)

    def __len__(self):
        return len(self._delegatee)