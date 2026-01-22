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
def argument_spec_schema(for_collection):
    any_string_types = Any(*string_types)
    schema = {any_string_types: {'type': Any(is_callable, *argument_spec_types), 'elements': Any(*argument_spec_types), 'default': object, 'fallback': Any((is_callable, list_string_types), [is_callable, list_string_types]), 'choices': Any([object], (object,)), 'required': bool, 'no_log': bool, 'aliases': Any(list_string_types, tuple(list_string_types)), 'apply_defaults': bool, 'removed_in_version': version(for_collection), 'removed_at_date': date(), 'removed_from_collection': collection_name, 'options': Self, 'deprecated_aliases': Any([All(Any({Required('name'): Any(*string_types), Required('date'): date(), Required('collection_name'): collection_name}, {Required('name'): Any(*string_types), Required('version'): version(for_collection), Required('collection_name'): collection_name}), partial(check_removal_version, version_field='version', collection_name_field='collection_name', error_code='invalid-removal-version'))])}}
    schema[any_string_types].update(argument_spec_modifiers)
    schemas = All(schema, Schema({any_string_types: no_required_with_default}), Schema({any_string_types: elements_with_list}), Schema({any_string_types: options_with_apply_defaults}), Schema({any_string_types: option_deprecation}))
    return Schema(schemas)