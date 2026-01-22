from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.urls import fetch_url
class Pushover(object):
    """ Instantiates a pushover object, use it to send notifications """
    base_uri = 'https://api.pushover.net'

    def __init__(self, module, user, token):
        self.module = module
        self.user = user
        self.token = token

    def run(self, priority, msg, title, device):
        """ Do, whatever it is, we do. """
        url = '%s/1/messages.json' % self.base_uri
        options = dict(user=self.user, token=self.token, priority=priority, message=msg)
        if title is not None:
            options = dict(options, title=title)
        if device is not None:
            options = dict(options, device=device)
        data = urlencode(options)
        headers = {'Content-type': 'application/x-www-form-urlencoded'}
        r, info = fetch_url(self.module, url, method='POST', data=data, headers=headers)
        if info['status'] != 200:
            raise Exception(info)
        return r.read()