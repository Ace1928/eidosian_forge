from __future__ import annotations
import datetime
import os
import re
import sys
from functools import partial
import yaml
from voluptuous import All, Any, MultipleInvalid, PREVENT_EXTRA
from voluptuous import Required, Schema, Invalid
from voluptuous.humanize import humanize_error
from ansible.module_utils.compat.version import StrictVersion, LooseVersion
from ansible.module_utils.six import string_types
from ansible.utils.collection_loader import AnsibleCollectionRef
from ansible.utils.version import SemanticVersion
def any_value(value):
    """Accepts anything."""
    return value