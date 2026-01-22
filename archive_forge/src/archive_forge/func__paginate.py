import json
import itertools
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import (
from libcloud.utils.misc import reverse_dict, merge_valid_keys
from libcloud.common.base import JsonResponse, ConnectionKey, ConnectionUserAndKey
from libcloud.common.types import LibcloudError, InvalidCredsError
def _paginate(self, get_page, page_size):
    for page in itertools.count(start=1):
        params = {'page': page, 'per_page': page_size}
        response, items = get_page(params)
        yield from items
        if self._is_last_page(response):
            break
        if len(items) < page_size:
            break