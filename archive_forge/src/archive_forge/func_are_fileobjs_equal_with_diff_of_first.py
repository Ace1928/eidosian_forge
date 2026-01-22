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
def are_fileobjs_equal_with_diff_of_first(f1, f2, size, diff, max_file_size_for_diff, container_path):
    if diff is None:
        return are_fileobjs_equal(f1, f2)
    if size > max_file_size_for_diff > 0:
        diff['dst_larger'] = max_file_size_for_diff
        return are_fileobjs_equal(f1, f2)
    is_equal, content = are_fileobjs_equal_read_first(f1, f2)
    if is_binary(content):
        diff['dst_binary'] = 1
    else:
        diff['before_header'] = container_path
        diff['before'] = to_text(content)
    return is_equal