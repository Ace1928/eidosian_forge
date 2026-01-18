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
def version_added(v, error_code='version-added-invalid', accept_historical=False):
    if 'version_added' in v:
        version_added = v.get('version_added')
        if isinstance(version_added, string_types):
            if v.get('version_added_collection') == 'ansible.builtin':
                if version_added == 'historical' and accept_historical:
                    return v
                try:
                    version = StrictVersion()
                    version.parse(version_added)
                except ValueError as exc:
                    raise _add_ansible_error_code(Invalid('version_added (%r) is not a valid ansible-core version: %s' % (version_added, exc)), error_code=error_code)
            else:
                try:
                    version = SemanticVersion()
                    version.parse(version_added)
                    if version.major != 0 and version.patch != 0:
                        raise _add_ansible_error_code(Invalid('version_added (%r) must be a major or minor release, not a patch release (see specification at https://semver.org/)' % (version_added,)), error_code='version-added-must-be-major-or-minor')
                except ValueError as exc:
                    raise _add_ansible_error_code(Invalid('version_added (%r) is not a valid collection version (see specification at https://semver.org/): %s' % (version_added, exc)), error_code=error_code)
    elif 'version_added_collection' in v:
        raise Invalid('version_added_collection cannot be specified without version_added')
    return v