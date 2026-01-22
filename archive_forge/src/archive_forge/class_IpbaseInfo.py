from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.six.moves.urllib.parse import urlencode
class IpbaseInfo(object):

    def __init__(self, module):
        self.module = module

    def _get_url_data(self, url):
        response, info = fetch_url(self.module, url, force=True, timeout=10, headers={'Accept': 'application/json', 'User-Agent': USER_AGENT})
        if info['status'] != 200:
            self.module.fail_json(msg='The API request to ipbase.com returned an error status code {0}'.format(info['status']))
        else:
            try:
                content = response.read()
                result = self.module.from_json(content.decode('utf8'))
            except ValueError:
                self.module.fail_json(msg='Failed to parse the ipbase.com response: {0} {1}'.format(url, content))
            else:
                return result

    def info(self):
        ip = self.module.params['ip']
        apikey = self.module.params['apikey']
        hostname = self.module.params['hostname']
        language = self.module.params['language']
        url = BASE_URL
        params = {}
        if ip:
            params['ip'] = ip
        if apikey:
            params['apikey'] = apikey
        if hostname:
            params['hostname'] = 1
        if language:
            params['language'] = language
        if params:
            url += '?' + urlencode(params)
        return self._get_url_data(url)