from __future__ import (absolute_import, division, print_function)
import json
from ansible.plugins.lookup import LookupBase
from ansible.module_utils.urls import open_url
class Etcd:

    def __init__(self, url, version, validate_certs):
        self.url = url
        self.version = version
        self.baseurl = '%s/%s/keys' % (self.url, self.version)
        self.validate_certs = validate_certs

    def _parse_node(self, node):
        path = {}
        if node.get('dir', False):
            for n in node.get('nodes', []):
                path[n['key'].split('/')[-1]] = self._parse_node(n)
        else:
            path = node['value']
        return path

    def get(self, key):
        url = '%s/%s?recursive=true' % (self.baseurl, key)
        data = None
        value = {}
        try:
            r = open_url(url, validate_certs=self.validate_certs)
            data = r.read()
        except Exception:
            return None
        try:
            item = json.loads(data)
            if self.version == 'v1':
                if 'value' in item:
                    value = item['value']
            elif 'node' in item:
                value = self._parse_node(item['node'])
            if 'errorCode' in item:
                value = 'ENOENT'
        except Exception:
            raise
        return value