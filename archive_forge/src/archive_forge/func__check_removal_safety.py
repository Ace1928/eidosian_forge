from __future__ import absolute_import, division, print_function
import abc
import bz2
import glob
import gzip
import io
import os
import re
import shutil
import tarfile
import zipfile
from fnmatch import fnmatch
from sys import version_info
from traceback import format_exc
from zlib import crc32
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_bytes, to_native
from ansible.module_utils import six
def _check_removal_safety(self):
    for path in self.paths:
        if os.path.isdir(path) and self.destination.startswith(os.path.join(path, b'')):
            self.module.fail_json(path=b', '.join(self.paths), msg='Error, created archive can not be contained in source paths when remove=true')