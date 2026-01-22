from __future__ import (absolute_import, division, print_function)
import sys
from contextlib import contextmanager
from ansible.template import Templar
from ansible.vars.manager import VariableManager
from ansible.plugins.callback.default import CallbackModule as Default
from ansible.module_utils.common.text.converters import to_text
class DummyStdout(object):

    def flush(self):
        pass

    def write(self, b):
        pass

    def writelines(self, l):
        pass