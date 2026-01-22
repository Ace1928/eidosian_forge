import json
import itertools
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import (
from libcloud.utils.misc import reverse_dict, merge_valid_keys
from libcloud.common.base import JsonResponse, ConnectionKey, ConnectionUserAndKey
from libcloud.common.types import LibcloudError, InvalidCredsError
def _is_last_page(self, response):
    try:
        result_info = response.object['result_info']
        last_page = result_info['total_pages']
        current_page = result_info['page']
    except KeyError:
        return False
    return current_page == last_page