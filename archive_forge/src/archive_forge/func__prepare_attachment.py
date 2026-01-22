from __future__ import absolute_import, division, print_function
import base64
import binascii
import json
import mimetypes
import os
import random
import string
import traceback
from ansible_collections.community.general.plugins.module_utils.module_helper import StateModuleHelper, cause_changes
from ansible.module_utils.six.moves.urllib.request import pathname2url
from ansible.module_utils.common.text.converters import to_text, to_bytes, to_native
from ansible.module_utils.urls import fetch_url
@staticmethod
def _prepare_attachment(filename, content=None, mime_type=None):

    def escape_quotes(s):
        return s.replace('"', '\\"')
    boundary = ''.join((random.choice(string.digits + string.ascii_letters) for dummy in range(30)))
    name = to_native(os.path.basename(filename))
    if not mime_type:
        try:
            mime_type = mimetypes.guess_type(filename or '', strict=False)[0] or 'application/octet-stream'
        except Exception:
            mime_type = 'application/octet-stream'
    main_type, sep, sub_type = mime_type.partition('/')
    if not content and filename:
        with open(to_bytes(filename, errors='surrogate_or_strict'), 'rb') as f:
            content = f.read()
    else:
        try:
            content = base64.b64decode(content)
        except binascii.Error as e:
            raise Exception('Unable to base64 decode file content: %s' % e)
    lines = ['--{0}'.format(boundary), 'Content-Disposition: form-data; name="file"; filename={0}'.format(escape_quotes(name)), 'Content-Type: {0}'.format('{0}/{1}'.format(main_type, sub_type)), '', to_text(content), '--{0}--'.format(boundary), '']
    return ('multipart/form-data; boundary={0}'.format(boundary), '\r\n'.join(lines))