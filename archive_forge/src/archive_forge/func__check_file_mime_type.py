from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six.moves.urllib import parse as urllib_parse
from mimetypes import MimeTypes
import os
import json
import traceback
@staticmethod
def _check_file_mime_type(path):
    mime = MimeTypes()
    return mime.guess_type(path)