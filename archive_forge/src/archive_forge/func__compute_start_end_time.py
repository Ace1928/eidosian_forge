from __future__ import absolute_import, division, print_function
import datetime
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
def _compute_start_end_time(self, hours, minutes):
    now = datetime.datetime.utcnow()
    later = now + datetime.timedelta(hours=int(hours), minutes=int(minutes))
    start = now.strftime('%Y-%m-%dT%H:%M:%SZ')
    end = later.strftime('%Y-%m-%dT%H:%M:%SZ')
    return (start, end)