import io
import logging
import random
import fixtures
from openstackclient import shell
from oslotest import base
from placement.tests.functional.fixtures import capture
from placement.tests.functional.fixtures import placement
import simplejson as json
def trait_delete(self, name):
    cmd = 'trait delete %s' % name
    self.openstack(cmd)