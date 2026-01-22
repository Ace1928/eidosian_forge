import base64
import warnings
from libcloud.pricing import get_size_price
from libcloud.utils.py3 import ET, b, next, httplib, parse_qs, urlparse
from libcloud.utils.xml import findall
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.iso8601 import parse_date
from libcloud.common.openstack import (
from libcloud.utils.networking import is_public_subnet
from libcloud.common.exceptions import BaseHTTPError
def _to_quota_set(self, obj):
    res = OpenStack_2_QuotaSet(id=obj['id'], cores=obj['cores'], instances=obj['instances'], key_pairs=obj['key_pairs'], metadata_items=obj['metadata_items'], ram=obj['ram'], server_groups=obj['server_groups'], server_group_members=obj['server_group_members'], fixed_ips=obj.get('fixed_ips', None), floating_ips=obj.get('floating_ips', None), networks=obj.get('networks', None), security_group_rules=obj.get('security_group_rules', None), security_groups=obj.get('security_groups', None), injected_file_content_bytes=obj.get('injected_file_content_bytes', None), injected_file_path_bytes=obj.get('injected_file_path_bytes', None), injected_files=obj.get('injected_files', None), driver=self.connection.driver)
    return res