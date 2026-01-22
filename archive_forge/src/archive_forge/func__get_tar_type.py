from __future__ import absolute_import, division, print_function
import binascii
import codecs
import datetime
import fnmatch
import grp
import os
import platform
import pwd
import re
import stat
import time
import traceback
from functools import partial
from zipfile import ZipFile
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.urls import fetch_file
def _get_tar_type(self):
    cmd = [self.cmd_path, '--version']
    rc, out, err = self.module.run_command(cmd)
    tar_type = None
    if out.startswith('bsdtar'):
        tar_type = 'bsd'
    elif out.startswith('tar') and 'GNU' in out:
        tar_type = 'gnu'
    return tar_type