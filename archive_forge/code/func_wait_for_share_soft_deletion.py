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
def wait_for_share_soft_deletion(self, share_id, microversion=None):
    body = self.get_share(share_id, microversion=microversion)
    is_soft_deleted = body['is_soft_deleted']
    start = int(time.time())
    while is_soft_deleted == 'False':
        time.sleep(self.build_interval)
        body = self.get_share(share_id, microversion=microversion)
        is_soft_deleted = body['is_soft_deleted']
        if is_soft_deleted == 'True':
            return
        if int(time.time()) - start >= self.build_timeout:
            message = 'Share %(share_id)s failed to be soft deleted within the required time %(build_timeout)s.' % {'share_id': share_id, 'build_timeout': self.build_timeout}
            raise tempest_lib_exc.TimeoutException(message)