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
@_unroll_iterator
def _ansible_finalize(thing):
    """A custom finalize function for jinja2, which prevents None from being
    returned. This avoids a string of ``"None"`` as ``None`` has no
    importance in YAML.

    The function is decorated with ``_unroll_iterator`` so that users are not
    required to explicitly use ``|list`` to unroll a generator. This only
    affects the scenario where the final result of templating
    is a generator, e.g. ``range``, ``dict.items()`` and so on. Filters
    which can produce a generator in the middle of a template are already
    wrapped with ``_unroll_generator`` in ``JinjaPluginIntercept``.
    """
    return thing if _fail_on_undefined(thing) is not None else ''