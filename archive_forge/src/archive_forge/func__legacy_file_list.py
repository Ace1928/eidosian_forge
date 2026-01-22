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
def _legacy_file_list(self):
    rc, out, err = self.module.run_command([self.cmd_path, '-v', self.src])
    if rc:
        self.module.debug(err)
        raise UnarchiveError('Neither python zipfile nor unzip can read %s' % self.src)
    for line in out.splitlines()[3:-2]:
        fields = line.split(None, 7)
        self._files_in_archive.append(fields[7])
        self._infodict[fields[7]] = int(fields[6])