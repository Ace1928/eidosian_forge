from __future__ import (absolute_import, division, print_function)
import errno
import datetime
import functools
import os
import tarfile
import tempfile
from collections.abc import MutableSequence
from shutil import rmtree
from ansible import context
from ansible.errors import AnsibleError, AnsibleParserError
from ansible.galaxy.api import GalaxyAPI
from ansible.galaxy.user_agent import user_agent
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.common.yaml import yaml_dump, yaml_load
from ansible.module_utils.compat.version import LooseVersion
from ansible.module_utils.urls import open_url
from ansible.playbook.role.requirement import RoleRequirement
from ansible.utils.display import Display
from ansible.utils.path import is_subpath, unfrackpath
@functools.cache
def _check_working_data_filter() -> bool:
    """
    Check if tarfile.data_filter implementation is working
    for the current Python version or not
    """
    ret = False
    if hasattr(tarfile, 'data_filter'):
        ti = tarfile.TarInfo('docs/README.md')
        ti.type = tarfile.SYMTYPE
        ti.linkname = '../README.md'
        try:
            tarfile.data_filter(ti, '/foo')
        except tarfile.LinkOutsideDestinationError:
            pass
        else:
            ret = True
    return ret