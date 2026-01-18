from __future__ import (absolute_import, division, print_function)
import json
from ansible.plugins.httpapi import HttpApiBase
from ansible.module_utils.basic import to_text
from ansible.module_utils.six.moves import urllib
import re
from datetime import datetime
def update_system_version(self):
    """
        retrieve the system status of fortigate device
        """
    self.log('checking system_version')
    check_system_status = self._conn.get_option('check_system_status') if 'check_system_status' in self._conn._options else True
    if not check_system_status or self._system_version:
        return
    url = '/api/v2/monitor/system/status?vdom=root'
    status, result = self.send_request(url=url)
    result_json = json.loads(result)
    self._system_version = result_json.get('version', 'undefined')
    self.log('system version: %s' % self._system_version)
    self.log('ansible version: %s' % self._ansible_fos_version)