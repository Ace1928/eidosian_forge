from __future__ import (absolute_import, division, print_function)
import codecs
import csv
from collections.abc import MutableSequence
from ansible.errors import AnsibleError, AnsibleAssertionError
from ansible.parsing.splitter import parse_kv
from ansible.plugins.lookup import LookupBase
from ansible.module_utils.six import PY2
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text

    A CSV reader which will iterate over lines in the CSV file "f",
    which is encoded in the given encoding.
    