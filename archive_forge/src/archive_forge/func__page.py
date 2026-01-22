import copy
from oslo_serialization import jsonutils
from urllib import parse
from saharaclient._i18n import _
def _page(self, url, response_key, limit=None):
    resp = self.api.get(url)
    if resp.status_code == 200:
        result = get_json(resp)
        data = result[response_key]
        meta = result.get('markers')
        next, prev = (None, None)
        if meta:
            prev = meta.get('prev')
            next = meta.get('next')
        li = [self.resource_class(self, res) for res in data]
        return Page(li, prev, next, limit)
    else:
        self._raise_api_exception(resp)