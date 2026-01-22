from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six.moves.urllib import parse as urllib_parse
from mimetypes import MimeTypes
import os
import json
import traceback
def append_ssl_certs(self):
    ssl_options = {}
    if self.cafile:
        ssl_options['cafile'] = self.cafile
    if self.certfile:
        ssl_options['certfile'] = self.certfile
    if self.keyfile:
        ssl_options['keyfile'] = self.keyfile
    self.url = self.url + '?ssl_options=' + urllib_parse.quote(json.dumps(ssl_options))