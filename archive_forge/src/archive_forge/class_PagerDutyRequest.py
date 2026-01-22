from __future__ import absolute_import, division, print_function
import datetime
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
class PagerDutyRequest(object):

    def __init__(self, module, name, user, token):
        self.module = module
        self.name = name
        self.user = user
        self.token = token
        self.headers = {'Content-Type': 'application/json', 'Authorization': self._auth_header(), 'Accept': 'application/vnd.pagerduty+json;version=2'}

    def ongoing(self, http_call=fetch_url):
        url = 'https://api.pagerduty.com/maintenance_windows?filter=ongoing'
        headers = dict(self.headers)
        response, info = http_call(self.module, url, headers=headers)
        if info['status'] != 200:
            self.module.fail_json(msg='failed to lookup the ongoing window: %s' % info['msg'])
        json_out = self._read_response(response)
        return (False, json_out, False)

    def create(self, requester_id, service, hours, minutes, desc, http_call=fetch_url):
        if not requester_id:
            self.module.fail_json(msg='requester_id is required when maintenance window should be created')
        url = 'https://api.pagerduty.com/maintenance_windows'
        headers = dict(self.headers)
        headers.update({'From': requester_id})
        start, end = self._compute_start_end_time(hours, minutes)
        services = self._create_services_payload(service)
        request_data = {'maintenance_window': {'start_time': start, 'end_time': end, 'description': desc, 'services': services}}
        data = json.dumps(request_data)
        response, info = http_call(self.module, url, data=data, headers=headers, method='POST')
        if info['status'] != 201:
            self.module.fail_json(msg='failed to create the window: %s' % info['msg'])
        json_out = self._read_response(response)
        return (False, json_out, True)

    def _create_services_payload(self, service):
        if isinstance(service, list):
            return [{'id': s, 'type': 'service_reference'} for s in service]
        else:
            return [{'id': service, 'type': 'service_reference'}]

    def _compute_start_end_time(self, hours, minutes):
        now = datetime.datetime.utcnow()
        later = now + datetime.timedelta(hours=int(hours), minutes=int(minutes))
        start = now.strftime('%Y-%m-%dT%H:%M:%SZ')
        end = later.strftime('%Y-%m-%dT%H:%M:%SZ')
        return (start, end)

    def absent(self, window_id, http_call=fetch_url):
        url = 'https://api.pagerduty.com/maintenance_windows/' + window_id
        headers = dict(self.headers)
        response, info = http_call(self.module, url, headers=headers, method='DELETE')
        if info['status'] != 204:
            self.module.fail_json(msg='failed to delete the window: %s' % info['msg'])
        json_out = self._read_response(response)
        return (False, json_out, True)

    def _auth_header(self):
        return 'Token token=%s' % self.token

    def _read_response(self, response):
        try:
            return json.loads(response.read())
        except Exception:
            return ''