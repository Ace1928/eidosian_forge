from __future__ import absolute_import, division, print_function
import base64
import io
import os
import stat
import traceback
from ansible.module_utils._text import to_bytes, to_native, to_text
from ansible_collections.community.docker.plugins.module_utils._api.errors import APIError, DockerException, NotFound
from ansible_collections.community.docker.plugins.module_utils.common_api import (
from ansible_collections.community.docker.plugins.module_utils.copy import (
from ansible_collections.community.docker.plugins.module_utils._scramble import generate_insecure_key, scramble
def add_other_diff(diff, in_path, member):
    if diff is None:
        return
    diff['before_header'] = in_path
    if member.isdir():
        diff['before'] = '(directory)'
    elif member.issym() or member.islnk():
        diff['before'] = member.linkname
    elif member.ischr():
        diff['before'] = '(character device)'
    elif member.isblk():
        diff['before'] = '(block device)'
    elif member.isfifo():
        diff['before'] = '(fifo)'
    elif member.isdev():
        diff['before'] = '(device)'
    elif member.isfile():
        raise DockerUnexpectedError('should not be a regular file')
    else:
        diff['before'] = '(unknown filesystem object)'