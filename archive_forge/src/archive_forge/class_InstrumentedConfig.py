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
class InstrumentedConfig(config.Config):
    """An instrumented config that supplies stubs for template methods."""

    def __init__(self):
        super().__init__()
        self._calls = []
        self._signatures = config.CHECK_NEVER
        self._change_editor = 'vimdiff -fo {new_path} {old_path}'

    def _get_user_id(self):
        self._calls.append('_get_user_id')
        return 'Robert Collins <robert.collins@example.org>'

    def _get_signature_checking(self):
        self._calls.append('_get_signature_checking')
        return self._signatures

    def _get_change_editor(self):
        self._calls.append('_get_change_editor')
        return self._change_editor