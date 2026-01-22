from __future__ import (absolute_import, division, print_function)
import base64
import datetime
import io
import json
import os
import os.path
import shutil
import stat
import tarfile
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.six import raise_from
from ansible_collections.community.docker.plugins.module_utils._api.errors import APIError, NotFound
def _stream_generator_to_fileobj(stream):
    """Given a generator that generates chunks of bytes, create a readable buffered stream."""
    raw = _RawGeneratorFileobj(stream)
    return io.BufferedReader(raw)