from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
class BackendServiceCdnpolicy(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({'cacheKeyPolicy': BackendServiceCachekeypolicy(self.request.get('cache_key_policy', {}), self.module).to_request(), 'signedUrlCacheMaxAgeSec': self.request.get('signed_url_cache_max_age_sec'), 'defaultTtl': self.request.get('default_ttl'), 'maxTtl': self.request.get('max_ttl'), 'clientTtl': self.request.get('client_ttl'), 'negativeCaching': self.request.get('negative_caching'), 'negativeCachingPolicy': BackendServiceNegativecachingpolicyArray(self.request.get('negative_caching_policy', []), self.module).to_request(), 'cacheMode': self.request.get('cache_mode'), 'serveWhileStale': self.request.get('serve_while_stale')})

    def from_response(self):
        return remove_nones_from_dict({'cacheKeyPolicy': BackendServiceCachekeypolicy(self.request.get('cacheKeyPolicy', {}), self.module).from_response(), 'signedUrlCacheMaxAgeSec': self.request.get('signedUrlCacheMaxAgeSec'), 'defaultTtl': self.request.get('defaultTtl'), 'maxTtl': self.request.get('maxTtl'), 'clientTtl': self.request.get('clientTtl'), 'negativeCaching': self.request.get('negativeCaching'), 'negativeCachingPolicy': BackendServiceNegativecachingpolicyArray(self.request.get('negativeCachingPolicy', []), self.module).from_response(), 'cacheMode': self.request.get('cacheMode'), 'serveWhileStale': self.request.get('serveWhileStale')})