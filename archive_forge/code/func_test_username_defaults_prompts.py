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
def test_username_defaults_prompts(self):
    self._check_default_username_prompt('FTP %(host)s username: ', 'ftp')
    self._check_default_username_prompt('FTP %(host)s:%(port)d username: ', 'ftp', port=10020)
    self._check_default_username_prompt('SSH %(host)s:%(port)d username: ', 'ssh', port=12345)