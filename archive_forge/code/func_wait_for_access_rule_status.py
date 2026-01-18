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
def wait_for_access_rule_status(self, share_id, access_id, state='active', microversion=None, is_snapshot=False):
    access = self.get_access(share_id, access_id, microversion=microversion, is_snapshot=is_snapshot)
    start = int(time.time())
    while access['state'] != state:
        time.sleep(self.build_interval)
        access = self.get_access(share_id, access_id, microversion=microversion, is_snapshot=is_snapshot)
        if access['state'] == state:
            return
        elif access['state'] == 'error':
            raise exceptions.AccessRuleCreateErrorException(access=access_id)
        if int(time.time()) - start >= self.build_timeout:
            message = 'Access rule %(access)s failed to reach %(state)s state within the required time (%(build_timeout)s s).' % {'access': access_id, 'state': state, 'build_timeout': self.build_timeout}
            raise tempest_lib_exc.TimeoutException(message)