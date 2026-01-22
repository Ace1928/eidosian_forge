import json
import time
from tempest.lib.cli import base
from tempest.lib.common.utils import data_utils
from manilaclient import config
def _wait_for_object_status(self, object_name, object_id, status, timeout=CONF.build_timeout, interval=CONF.build_interval):
    """Waits for a object to reach a given status."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if status == self.openstack('%(obj)s show -c status -f value %(id)s' % {'obj': object_name, 'id': object_id}).rstrip():
            break
        time.sleep(interval)
    else:
        self.fail('%s %s did not reach status %s after %d seconds.' % (object_name, object_id, status, timeout))