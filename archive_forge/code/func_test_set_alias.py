import os
import sys
import threading
from io import BytesIO
from textwrap import dedent
import configobj
from testtools import matchers
from .. import (bedding, branch, config, controldir, diff, errors, lock,
from .. import registry as _mod_registry
from .. import tests, trace
from .. import transport as _mod_transport
from .. import ui, urlutils
from ..bzr import remote
from ..transport import remote as transport_remote
from . import features, scenarios, test_server
def test_set_alias(self):
    my_config = config.GlobalConfig()
    alias_value = 'commit --strict'
    my_config.set_alias('commit', alias_value)
    new_config = config.GlobalConfig()
    self.assertEqual(alias_value, new_config.get_alias('commit'))