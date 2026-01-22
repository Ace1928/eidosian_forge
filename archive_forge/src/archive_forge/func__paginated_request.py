import json
from typing import Any, Dict, List, Optional
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import (
from libcloud.utils.py3 import urlencode
from libcloud.common.vultr import (
def _paginated_request(self, url: str, key: str, params: Optional[Dict[str, Any]]=None) -> List[Any]:
    """Perform multiple calls to get the full list of items when
        the API responses are paginated.

        :param url: API endpoint
        :type url: ``str``

        :param key: Result object key
        :type key: ``str``

        :param params: Request parameters
        :type params: ``dict``

        :return: ``list`` of API response objects
        :rtype: ``list``
        """
    params = params if params is not None else {}
    resp = self.connection.request(url, params=params).object
    data = list(resp.get(key, []))
    objects = data
    while True:
        next_page = resp['meta']['links']['next']
        if next_page:
            params['cursor'] = next_page
            resp = self.connection.request(url, params=params).object
            data = list(resp.get(key, []))
            objects.extend(data)
        else:
            return objects