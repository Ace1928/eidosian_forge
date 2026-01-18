import os
import sys
from .... import (bedding, controldir, errors, osutils, revisionspec, tests,
from ....tests import features, per_branch, per_transport
from .. import cmds
def test_set_push_location(self):
    conf = self.get_branch().get_config_stack()
    conf.set('upload_location', 'foo')
    self.assertEqual('foo', conf.get('upload_location'))