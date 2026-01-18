from __future__ import absolute_import, division, print_function
import os
import re
import shutil
import tempfile
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.parse import urlsplit
from ansible.module_utils.compat.datetime import utcnow, utcfromtimestamp
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.urls import fetch_url, url_argument_spec
def url_filename(url):
    fn = os.path.basename(urlsplit(url)[2])
    if fn == '':
        return 'index.html'
    return fn