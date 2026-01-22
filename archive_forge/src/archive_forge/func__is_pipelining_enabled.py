from __future__ import (absolute_import, division, print_function)
import base64
import json
import os
import random
import re
import shlex
import stat
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Sequence
from ansible import constants as C
from ansible.errors import AnsibleError, AnsibleConnectionFailure, AnsibleActionSkip, AnsibleActionFail, AnsibleAuthenticationFailure
from ansible.executor.module_common import modify_module
from ansible.executor.interpreter_discovery import discover_interpreter, InterpreterDiscoveryRequiredError
from ansible.module_utils.common.arg_spec import ArgumentSpecValidator
from ansible.module_utils.errors import UnsupportedError
from ansible.module_utils.json_utils import _filter_non_json_lines
from ansible.module_utils.six import binary_type, string_types, text_type
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.parsing.utils.jsonify import jsonify
from ansible.release import __version__
from ansible.utils.collection_loader import resource_from_fqcr
from ansible.utils.display import Display
from ansible.utils.unsafe_proxy import wrap_var, AnsibleUnsafeText
from ansible.vars.clean import remove_internal_keys
from ansible.utils.plugin_docs import get_versioned_doclink
def _is_pipelining_enabled(self, module_style, wrap_async=False):
    """
        Determines if we are required and can do pipelining
        """
    try:
        is_enabled = self._connection.get_option('pipelining')
    except (KeyError, AttributeError, ValueError):
        is_enabled = self._play_context.pipelining
    always_pipeline = self._connection.always_pipeline_modules
    become_exception = (self._connection.become.name if self._connection.become else '') != 'su'
    conditions = [self._connection.has_pipelining, is_enabled or always_pipeline, module_style == 'new', not C.DEFAULT_KEEP_REMOTE_FILES, not wrap_async or always_pipeline, become_exception]
    return all(conditions)