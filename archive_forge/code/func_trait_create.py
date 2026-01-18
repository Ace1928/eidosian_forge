import io
import logging
import random
import fixtures
from openstackclient import shell
from oslotest import base
from placement.tests.functional.fixtures import capture
from placement.tests.functional.fixtures import placement
import simplejson as json
def trait_create(self, name):
    cmd = 'trait create %s' % name
    self.openstack(cmd)

    def cleanup():
        try:
            self.trait_delete(name)
        except CommandException as exc:
            err_message = str(exc).lower()
            if 'http 404' not in err_message:
                raise
    self.addCleanup(cleanup)