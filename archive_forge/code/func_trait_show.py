import io
import logging
import random
import fixtures
from openstackclient import shell
from oslotest import base
from placement.tests.functional.fixtures import capture
from placement.tests.functional.fixtures import placement
import simplejson as json
def trait_show(self, name):
    cmd = 'trait show %s' % name
    return self.openstack(cmd, use_json=True)