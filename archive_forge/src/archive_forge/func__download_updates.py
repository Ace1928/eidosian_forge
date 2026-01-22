from __future__ import absolute_import, division, print_function
import hashlib
import io
import json
import os
import tempfile
from ansible.module_utils.basic import AnsibleModule, to_bytes
from ansible.module_utils.six.moves import http_cookiejar as cookiejar
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.urls import fetch_url, url_argument_spec
from ansible.module_utils.six import text_type, binary_type
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.jenkins import download_updates_file
def _download_updates(self):
    try:
        updates_file, download_updates = download_updates_file(self.params['updates_expiration'])
    except OSError as e:
        self.module.fail_json(msg='Cannot create temporal directory.', details=to_native(e))
    if download_updates:
        urls = self._get_update_center_urls()
        r = self._get_urls_data(urls, msg_status='Remote updates not found.', msg_exception='Updates download failed.')
        tmp_update_fd, tmp_updates_file = tempfile.mkstemp()
        os.write(tmp_update_fd, r.read())
        try:
            os.close(tmp_update_fd)
        except IOError as e:
            self.module.fail_json(msg='Cannot close the tmp updates file %s.' % tmp_updates_file, details=to_native(e))
    else:
        tmp_updates_file = updates_file
    try:
        f = io.open(tmp_updates_file, encoding='utf-8')
        dummy = f.readline()
        data = json.loads(f.readline())
    except IOError as e:
        self.module.fail_json(msg='Cannot open%s updates file.' % (' temporary' if tmp_updates_file != updates_file else ''), details=to_native(e))
    except Exception as e:
        self.module.fail_json(msg='Cannot load JSON data from the%s updates file.' % (' temporary' if tmp_updates_file != updates_file else ''), details=to_native(e))
    if tmp_updates_file != updates_file:
        self.module.atomic_move(tmp_updates_file, updates_file)
    if not data.get('plugins', {}).get(self.params['name']):
        self.module.fail_json(msg='Cannot find plugin data in the updates file.')
    return data['plugins'][self.params['name']]