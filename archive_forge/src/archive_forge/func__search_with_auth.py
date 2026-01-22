from math import ceil
from boto.compat import json, map, six
import requests
from boto.cloudsearchdomain.layer1 import CloudSearchDomainConnection
def _search_with_auth(self, params):
    return self.domain_connection.search(params.pop('q', ''), **params)