import time
from libcloud.compute.base import (
from libcloud.utils.iso8601 import parse_date
from libcloud.common.gridscale import GridscaleBaseDriver, GridscaleConnection
from libcloud.compute.providers import Provider
def _to_ip(self, data):
    extra_keys = ['create_time', 'current_price', 'name', 'relations', 'reverse_dns', 'status']
    extra = self._extract_values_to_dict(data=data, keys=extra_keys)
    ip = GridscaleIp(id=data['object_uuid'], family=data['family'], prefix=data['prefix'], create_time=data['create_time'], address=data['ip'], extra=extra)
    return ip