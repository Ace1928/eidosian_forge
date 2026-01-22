from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_text
from ansible_collections.community.grafana.plugins.module_utils.base import (
from ansible.module_utils.urls import basic_auth_header
class GrafanaNotificationChannelInterface(object):

    def __init__(self, module):
        self._module = module
        self.headers = {'Content-Type': 'application/json'}
        if module.params.get('grafana_api_key', None):
            self.headers['Authorization'] = 'Bearer %s' % module.params['grafana_api_key']
        else:
            self.headers['Authorization'] = basic_auth_header(module.params['url_username'], module.params['url_password'])
        self.grafana_url = clean_url(module.params.get('url'))

    def grafana_switch_organisation(self, grafana_url, org_id):
        r, info = fetch_url(self._module, '%s/api/user/using/%s' % (grafana_url, org_id), headers=self.headers, method='POST')
        if info['status'] != 200:
            raise GrafanaAPIException('Unable to switch to organization %s : %s' % (org_id, info))

    def grafana_create_notification_channel(self, data, payload):
        r, info = fetch_url(self._module, '%s/api/alert-notifications' % data['url'], data=json.dumps(payload), headers=self.headers, method='POST')
        if info['status'] == 200:
            return {'state': 'present', 'changed': True, 'channel': json.loads(to_text(r.read()))}
        else:
            raise GrafanaAPIException('Unable to create notification channel: %s' % info)

    def grafana_update_notification_channel(self, data, payload, before):
        r, info = fetch_url(self._module, '%s/api/alert-notifications/uid/%s' % (data['url'], data['uid']), data=json.dumps(payload), headers=self.headers, method='PUT')
        if info['status'] == 200:
            del before['created']
            del before['updated']
            channel = json.loads(to_text(r.read()))
            after = channel.copy()
            del after['created']
            del after['updated']
            if before == after:
                return {'changed': False, 'channel': channel}
            else:
                return {'changed': True, 'diff': {'before': before, 'after': after}, 'channel': channel}
        else:
            raise GrafanaAPIException('Unable to update notification channel %s : %s' % (data['uid'], info))

    def grafana_create_or_update_notification_channel(self, data):
        payload = grafana_notification_channel_payload(data)
        r, info = fetch_url(self._module, '%s/api/alert-notifications/uid/%s' % (data['url'], data['uid']), headers=self.headers)
        if info['status'] == 200:
            before = json.loads(to_text(r.read()))
            return self.grafana_update_notification_channel(data, payload, before)
        elif info['status'] == 404:
            return self.grafana_create_notification_channel(data, payload)
        else:
            raise GrafanaAPIException('Unable to get notification channel %s : %s' % (data['uid'], info))

    def grafana_delete_notification_channel(self, data):
        r, info = fetch_url(self._module, '%s/api/alert-notifications/uid/%s' % (data['url'], data['uid']), headers=self.headers, method='DELETE')
        if info['status'] == 200:
            return {'state': 'absent', 'changed': True}
        elif info['status'] == 404:
            return {'changed': False}
        else:
            raise GrafanaAPIException('Unable to delete notification channel %s : %s' % (data['uid'], info))