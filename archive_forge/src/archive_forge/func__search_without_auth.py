from math import ceil
from boto.compat import json, map, six
import requests
from boto.cloudsearchdomain.layer1 import CloudSearchDomainConnection
def _search_without_auth(self, params, api_version):
    url = 'http://%s/%s/search' % (self.endpoint, api_version)
    resp = self.session.get(url, params=params)
    return {'body': resp.content.decode('utf-8'), 'status_code': resp.status_code}