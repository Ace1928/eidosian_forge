import time
from libcloud.compute.base import (
from libcloud.utils.iso8601 import parse_date
from libcloud.common.gridscale import GridscaleBaseDriver, GridscaleConnection
from libcloud.compute.providers import Provider
def _to_key(self, data, name=None, sshkey=None):
    extra = {'uuid': data['object_uuid'], 'labels': data.get('labels', [])}
    name = data.get('name', name)
    sshkey = data.get('sshkey', sshkey)
    key = KeyPair(name=name, fingerprint=data['object_uuid'], public_key=sshkey, private_key=None, extra=extra, driver=self.connection.driver)
    return key