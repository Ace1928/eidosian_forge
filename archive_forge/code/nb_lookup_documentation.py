from __future__ import absolute_import, division, print_function
import os
import functools
from pprint import pformat
from ansible.errors import AnsibleError
from ansible.plugins.lookup import LookupBase
from ansible.parsing.splitter import parse_kv, split_args
from ansible.utils.display import Display
from ansible.module_utils.six import raise_from
from importlib.metadata import version

    LookupModule(LookupBase) is defined by Ansible
    