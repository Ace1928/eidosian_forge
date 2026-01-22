from __future__ import absolute_import, division, print_function
import errno
import traceback
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.docker.plugins.module_utils.common_api import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils._api.errors import DockerException
@staticmethod
def _extract_output_line(line, output):
    """
        Extract text line from stream output and, if found, adds it to output.
        """
    if 'stream' in line or 'status' in line:
        text_line = line.get('stream') or line.get('status') or ''
        output.extend(text_line.splitlines())