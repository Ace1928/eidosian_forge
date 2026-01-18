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
def wait_for_message(self, resource_id):
    """Waits until a message for a resource with given id exists"""
    start = int(time.time())
    message = None
    while not message:
        time.sleep(self.build_interval)
        for msg in self.list_messages():
            if msg['Resource ID'] == resource_id:
                return msg
        if int(time.time()) - start >= self.build_timeout:
            message = 'No message for resource with id %s was created in the required time (%s s).' % (resource_id, self.build_timeout)
            raise tempest_lib_exc.TimeoutException(message)