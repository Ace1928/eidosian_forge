import os
import time
from cinderclient.v3 import client as cinderclient
import fixtures
from glanceclient import client as glanceclient
from keystoneauth1.exceptions import discovery as discovery_exc
from keystoneauth1 import identity
from keystoneauth1 import session as ksession
from keystoneclient import client as keystoneclient
from keystoneclient import discover as keystone_discover
from neutronclient.v2_0 import client as neutronclient
import openstack.config
import openstack.config.exceptions
from oslo_utils import uuidutils
import tempest.lib.cli.base
import testtools
import novaclient
import novaclient.api_versions
from novaclient import base
import novaclient.client
from novaclient.v2 import networks
import novaclient.v2.shell
def wait_for_volume_status(self, volume, status, timeout=60, poll_interval=1):
    """Wait until volume reaches given status.

        :param volume: volume resource
        :param status: expected status of volume
        :param timeout: timeout in seconds
        :param poll_interval: poll interval in seconds
        """
    start_time = time.time()
    while time.time() - start_time < timeout:
        volume = self.cinder.volumes.get(volume.id)
        if volume.status == status:
            break
        time.sleep(poll_interval)
    else:
        self.fail('Volume %s did not reach status %s after %d s' % (volume.id, status, timeout))