import ast
import re
import time
from oslo_utils import strutils
from tempest.lib.cli import base
from tempest.lib.cli import output_parser
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions as tempest_lib_exc
from manilaclient.common import constants
from manilaclient import config
from manilaclient.tests.functional import exceptions
from manilaclient.tests.functional import utils
def wait_for_resource_status(self, resource_id, status, microversion=None, resource_type='share'):
    """Waits for a share to reach a given status."""
    get_func = getattr(self, 'get_' + resource_type)
    body = get_func(resource_id, microversion=microversion)
    share_status = body['status']
    start = int(time.time())
    while share_status != status:
        time.sleep(self.build_interval)
        body = get_func(resource_id, microversion=microversion)
        share_status = body['status']
        if share_status == status:
            return
        elif 'error' in share_status.lower():
            raise exceptions.ShareBuildErrorException(share=resource_id)
        if int(time.time()) - start >= self.build_timeout:
            message = 'Resource %(resource_id)s failed to reach %(status)s  status within the required time (%(build_timeout)s).' % {'resource_id': resource_id, 'status': status, 'build_timeout': self.build_timeout}
            raise tempest_lib_exc.TimeoutException(message)