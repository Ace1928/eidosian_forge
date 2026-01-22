from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.urls import fetch_url
class Rest(object):

    def __init__(self, module, headers, baseurl):
        self.module = module
        self.headers = headers
        self.baseurl = baseurl

    def _url_builder(self, path):
        if path[0] == '/':
            path = path[1:]
        return '%s/%s' % (self.baseurl, path)

    def send(self, method, path, data=None, headers=None):
        url = self._url_builder(path)
        data = self.module.jsonify(data)
        resp, info = fetch_url(self.module, url, data=data, headers=self.headers, method=method)
        return Response(resp, info)

    def get(self, path, data=None, headers=None):
        return self.send('GET', path, data, headers)

    def put(self, path, data=None, headers=None):
        return self.send('PUT', path, data, headers)

    def post(self, path, data=None, headers=None):
        return self.send('POST', path, data, headers)

    def patch(self, path, data=None, headers=None):
        return self.send('PATCH', path, data, headers)

    def delete(self, path, data=None, headers=None):
        return self.send('DELETE', path, data, headers)