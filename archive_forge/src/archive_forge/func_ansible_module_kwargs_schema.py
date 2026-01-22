from __future__ import annotations
import re
from ansible.module_utils.compat.version import StrictVersion
from functools import partial
from urllib.parse import urlparse
from voluptuous import ALLOW_EXTRA, PREVENT_EXTRA, All, Any, Invalid, Length, MultipleInvalid, Required, Schema, Self, ValueInvalid, Exclusive
from ansible.constants import DOCUMENTABLE_PLUGINS
from ansible.module_utils.six import string_types
from ansible.module_utils.common.collections import is_iterable
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.parsing.quoting import unquote
from ansible.utils.version import SemanticVersion
from ansible.release import __version__
from antsibull_docs_parser import dom
from antsibull_docs_parser.parser import parse, Context
from .utils import parse_isodate
def ansible_module_kwargs_schema(module_name, for_collection):
    schema = {'argument_spec': argument_spec_schema(for_collection), 'bypass_checks': bool, 'no_log': bool, 'check_invalid_arguments': Any(None, bool), 'add_file_common_args': bool, 'supports_check_mode': bool}
    if module_name.endswith(('_info', '_facts')):
        del schema['supports_check_mode']
        schema[Required('supports_check_mode')] = True
    schema.update(argument_spec_modifiers)
    return Schema(schema)